import { useEffect, useState } from 'preact/hooks'
import './app.css'

import { YumePDQ, default as init } from './pkg/yume_pdq'
import { ImageEditor } from './components/ImageEditor';

export function App() {
  const [yumePDQ, setYumePDQ] = useState<YumePDQ | null>(null);
  const [fatalError, setFatalError] = useState<string | null>(null); ``

  useEffect(() => {
    init().then(() => {
      setYumePDQ(new YumePDQ())
    }).catch((e) => {
      setFatalError(e.message)
    })
  }, [])

  return fatalError ? (
    <div>
      <h1>Fatal Error</h1>
      <p>{fatalError}</p>
    </div>
  ) : !yumePDQ ? (
    <div>
      <h1>Initializing...</h1>
    </div>
  ) : (
    <>
      <div>
        <h1>YumePDQ WASM demo</h1>
        <ImageEditor />
      </div>
    </>
  )
}

