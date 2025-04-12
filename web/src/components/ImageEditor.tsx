import { useEffect, useRef, useState } from "preact/hooks";
import { YumePDQ } from "../pkg/yume_pdq";


function toHex(array: Uint8Array) {
    return Array.from(array).map(b => b.toString(16).padStart(2, '0')).join('');
}

function toBinaryString(array: Uint8Array) {
    return Array.from(array).map(b => b.toString(2).padStart(8, '0')).join('');
}

function toBase64(array: Uint8Array) {
    return btoa(String.fromCharCode(...array));
}

function formatDihedral(dihedral: number[]) {
    let xtox = dihedral[0];
    let ytox = dihedral[1];
    let xtoy = dihedral[2];
    let ytoy = dihedral[3];
    return `[[${xtox}, ${ytox}], [${xtoy}, ${ytoy}]]`;
}

export function ImageEditor() {
    const [yumePDQ, _0] = useState<YumePDQ>(new YumePDQ());
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvasInitialized, setCanvasInitialized] = useState(false);
    const [inputImageData, _1] = useState<Float32Array>(new Float32Array(512 * 512));
    const [tmpHash, _3] = useState<Uint8Array>(new Uint8Array(32));
    const [currentHash, setCurrentHash] = useState<[number, number[], Uint8Array][]>([]);
    const [binarySelectionIndex, setBinarySelectionIndex] = useState<number | null>(null);
    const [benchmarkTime, setBenchmarkTime] = useState<number | null>(null);
    useEffect(() => {
        if (canvasRef.current && !canvasInitialized) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
                ctx.fillStyle = '#FEDFE1';
                ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                setCanvasInitialized(true);
            }
        }
    }, []);

    const performHash = () => {
        if (!canvasRef.current) {
            console.warn('Canvas not initialized');
            return;
        }
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) {
            console.warn('Context not initialized');
            return;
        }
        const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height, {
            colorSpace: 'srgb',
        });
        yumePDQ.cvt_rgba8_to_luma8f(new Uint8Array(imageData.data), inputImageData);
        let putputList: [number, number[], Uint8Array][] = [];
        yumePDQ.hash_luma8(inputImageData, tmpHash, 1, (quality: number, dihedral: Int8Array, output: Uint8Array) => {
            console.log(dihedral);
            putputList.push([quality, [...dihedral], output]);
            return true;
        });
        setCurrentHash(putputList);
    }

    const benchmark = () => {
        const start = performance.now();
        if (!canvasRef.current) {
            console.warn('Canvas not initialized');
            return;
        }
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) {
            console.warn('Context not initialized');
            return;
        }
        const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height, {
            colorSpace: 'srgb',
        });
        let results = [];
        for (let i = 0; i < 100; i++) {
            yumePDQ.cvt_rgba8_to_luma8f(new Uint8Array(imageData.data), inputImageData);
            yumePDQ.hash_luma8(inputImageData, tmpHash, 1, (quality: number, dihedral: Int8Array, output: Uint8Array) => {
                results.push([quality, [...dihedral], output]);
                return true;
            });
        }
        const end = performance.now();
        setBenchmarkTime((end - start) / 100);
    }

    const handleFileChange = (event: Event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const image = new Image();
                image.src = e.target?.result as string;
                image.onload = () => {
                    const canvas = canvasRef.current;
                    if (!canvas) {
                        console.warn('Canvas not initialized');
                        return;
                    }
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.fillStyle = 'black';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvas.width, canvas.height);
                    }
                }
            }
            reader.readAsDataURL(file);
        }
    }

    const showTrackingNote = currentHash.length > 0 && currentHash[0][0] < 0.1;

    return (
        <div className="canvas-container">
            <canvas ref={canvasRef} width={512} height={512} />
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <button onClick={performHash}>Hash</button>
            <button onClick={benchmark}>{benchmarkTime ? `${(benchmarkTime * 1000).toFixed(3)} us` : 'Benchmark'}</button>
            <div>
                <p>Kernel: {yumePDQ.kernel_ident}</p>
                {showTrackingNote && <p>If quality is unexpectedly low, it may be your browser randomized the canvas extract, please temporarily disable tracking protection and try again.</p>}

                <table>
                    <thead>
                        <tr>
                            <th>Dihedral</th>
                            <th>Quality</th>
                            <th>Base64</th>
                            <th>Hex</th>
                        </tr>
                    </thead>
                    <tbody>
                        {currentHash.map((currentHash) => <tr><td className="chop">{formatDihedral(currentHash[1])}</td><td className="chop">{currentHash[0]}</td><td className="chop">{toBase64(currentHash[2])}</td><td className="chop">{toHex(currentHash[2])}</td></tr>)}
                    </tbody>
                </table>

                <p>Binary: </p>
                <select onChange={(e) => setBinarySelectionIndex(parseInt((e.target as HTMLSelectElement).value) || null)}>
                    <option value="" onSelect={() => setBinarySelectionIndex(null)}>Select one transformation</option>
                    {currentHash.map((currentHash, i) => <option value={i} key={i} selected={i === binarySelectionIndex}>{formatDihedral(currentHash[1])}</option>)}
                </select>
                <p className="chop">{binarySelectionIndex !== null ? toBinaryString(currentHash[binarySelectionIndex][2]) : ''}</p>
            </div>
        </div >
    )
}