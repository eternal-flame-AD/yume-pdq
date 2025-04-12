use generic_array::{
    GenericArray,
    sequence::Unflatten,
    typenum::{B0, U32, U512, UInt},
};
use wasm_bindgen::prelude::*;

use crate::{
    PDQHashF,
    alignment::Align32,
    kernel::{
        Kernel, SquareGenericArrayExt, constants,
        type_traits::{DivisibleBy8, SquareOf},
    },
};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &JsValue);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

/// Visit all 8 dihedral hashes using a callback function.
pub static FLAG_ALL_DIHEDRALS: u32 = 1;
/// Receive the intermediate dihedral hashes.
pub static FLAG_RECEIVE_PDQF: u32 = 2;

#[wasm_bindgen(js_name = YumePDQ)]
#[derive(Default)]
/// A YumePDQ instance for WASM.
pub struct JSYumePDQ {
    #[cfg(all(feature = "portable-simd", target_feature = "simd128"))]
    inner: JsYumePDQInner<crate::kernel::portable_simd::PortableSimdF32Kernel<4>>,
    #[cfg(all(feature = "portable-simd", not(target_feature = "simd128")))]
    inner: JsYumePDQInner<crate::kernel::DefaultKernel>,
}

type Times4<T> = UInt<UInt<T, B0>, B0>;

#[wasm_bindgen(js_class = YumePDQ)]
impl JSYumePDQ {
    #[wasm_bindgen(constructor)]
    /// Create a new YumePDQ instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the identifier of the kernel.
    #[wasm_bindgen(getter)]
    pub fn kernel_ident(&self) -> String {
        self.inner.kernel.ident().to_string()
    }

    #[wasm_bindgen]
    /// Convert a 512x512x4 image to a 512x512 luma8 image.
    pub fn cvt_rgba8_to_luma8f(
        &mut self,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<bool, JsValue> {
        let input_array =
            GenericArray::<u8, Times4<<U512 as SquareOf>::Output>>::try_from_slice(input)
                .map_err(|_| "Input buffer is not 512x512x4".to_string())?;

        let output_array =
            GenericArray::<f32, <U512 as SquareOf>::Output>::try_from_mut_slice(output)
                .map_err(|_| "Output buffer is not 512x512".to_string())?;

        for i in 0..512 {
            let input_offset = 512 * 4 * i;
            let output_offset = 512 * i;

            let input_slice = GenericArray::<u8, Times4<U512>>::from_slice(
                &input_array[input_offset..input_offset + 512 * 4],
            )
            .unflatten();
            let output_slice = GenericArray::<f32, U512>::from_mut_slice(
                &mut output_array[output_offset..output_offset + 512],
            );

            self.inner.kernel.cvt_rgba8_to_luma8f::<{
                u32::from_ne_bytes(constants::RGB8_TO_LUMA8_TABLE_ITU[0].to_ne_bytes())
            }, {
                u32::from_ne_bytes(constants::RGB8_TO_LUMA8_TABLE_ITU[1].to_ne_bytes())
            }, {
                u32::from_ne_bytes(constants::RGB8_TO_LUMA8_TABLE_ITU[2].to_ne_bytes())
            }>(input_slice, output_slice);
        }
        Ok(true)
    }

    #[wasm_bindgen]
    /// Hash a 512x512 luma8 image (pre-casted to f32), write the result to the 32-byte output buffer.
    ///
    /// # Arguments
    ///
    /// - `input`: A pointer to the input image data.
    /// - `output`: A pointer to the output buffer.
    /// - `flags`: A bitmask of flags. e.g. [`FLAG_ALL_DIHEDRALS`] or [`FLAG_RECEIVE_PDQF`].
    /// - `callback`: An optional callback function. You must use a callback function to get the intermediate dihedral hashes.
    ///
    /// # Returns
    ///
    /// - `quality`: The quality of the hash of the original image.
    ///
    pub fn hash_luma8(
        &mut self,
        input: &[f32],
        output: &mut [u8],
        flags: u32,
        callback: Option<js_sys::Function>,
    ) -> Result<f32, JsValue> {
        use generic_array::sequence::Flatten;

        let input_array = GenericArray::<
            _,
            <<crate::kernel::SmartKernelConcreteType as Kernel>::InputDimension as SquareOf>::Output,
        >::try_from_slice(input)
        .map_err(|_| "Input buffer is not 512x512".to_string())?.unflatten_square_ref();

        let mut output_array = GenericArray::<_, U32>::try_from_mut_slice(output)
            .map_err(|_| "Output buffer is not 32 bytes".to_string())?
            .unflatten();

        let mut threshold = 0.0;

        let quality = crate::hash_get_threshold(
            &mut self.inner.kernel,
            input_array,
            &mut threshold,
            &mut output_array,
            &mut self.inner.buf1_input,
            &mut self.inner.buf1_tmp,
            &mut self.inner.buf1_pdqf,
        );

        if let Some(callback) = callback {
            if flags & FLAG_RECEIVE_PDQF != 0 {
                let pdqf_data = self
                    .inner
                    .buf1_pdqf
                    .flatten()
                    .as_slice()
                    .to_vec()
                    .into_boxed_slice();
                callback.call3(
                    &JsValue::NULL,
                    &JsValue::from_f64(quality as _),
                    &JsValue::from([1i8, 0, 0, 1].to_vec().into_boxed_slice()),
                    &JsValue::from(pdqf_data),
                )?;
            } else {
                callback.call3(
                    &JsValue::NULL,
                    &JsValue::from_f64(quality as _),
                    &JsValue::from([1i8, 0, 0, 1].to_vec().into_boxed_slice()),
                    &JsValue::from(
                        output_array
                            .flatten()
                            .as_slice()
                            .to_vec()
                            .into_boxed_slice(),
                    ),
                )?;
            }

            if flags & FLAG_ALL_DIHEDRALS != 0 {
                crate::visit_dihedrals(
                    &mut self.inner.kernel,
                    &mut self.inner.buf1_pdqf,
                    &mut output_array,
                    threshold,
                    |coords, _, (quality, pdqf, output)| {
                        let coords = coords.into_tuples();
                        let dihedral = [coords.0.0, coords.0.1, coords.1.0, coords.1.1]
                            .to_vec()
                            .into_boxed_slice();
                        if flags & FLAG_RECEIVE_PDQF != 0 {
                            let pdqf_data = pdqf.flatten().as_slice().to_vec().into_boxed_slice();
                            callback.call3(
                                &JsValue::NULL,
                                &JsValue::from_f64(quality as _),
                                &JsValue::from(dihedral),
                                &JsValue::from(pdqf_data),
                            )?;
                        } else {
                            callback.call3(
                                &JsValue::NULL,
                                &JsValue::from_f64(quality as _),
                                &JsValue::from(dihedral),
                                &JsValue::from(
                                    output.flatten().as_slice().to_vec().into_boxed_slice(),
                                ),
                            )?;
                        }

                        Ok::<(), JsValue>(())
                    },
                )?;
            }
        }

        Ok(quality)
    }

    #[wasm_bindgen(js_name = dispose)]
    /// Dispose of the instance.
    pub fn dispose(self) {
        drop(self);
    }
}

#[derive(Default)]
struct JsYumePDQInner<K: Kernel>
where
    <K as Kernel>::OutputDimension: DivisibleBy8,
{
    kernel: K,
    buf1_input: Align32<GenericArray<GenericArray<f32, K::Buffer1WidthX>, K::Buffer1LengthY>>,
    buf1_pdqf: Align32<PDQHashF<K::InternalFloat, K::OutputDimension>>,
    buf1_tmp: Align32<GenericArray<K::InternalFloat, K::Buffer1WidthX>>,
}
