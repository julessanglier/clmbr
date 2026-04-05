import { useEffect, useState, useCallback } from 'react'

const MOUNTAIN_FRAMES = [
  `
                .
               /\\
              /  \\
`,
  `
                .
               /\\
              /  \\
             /    \\
            /      \\
`,
  `
                .
               /\\
              /  \\
         .   /    \\
        /\\  /      \\
       /  \\/        \\
`,
  `
                .
               /\\
              /  \\
         .   /    \\   .
        /\\  /      \\ /\\
       /  \\/        X  \\
      /    \\       / \\  \\
`,
  `
                .
               /\\
              /  \\
         .   /    \\   .
        /\\  /      \\ /\\
    .  /  \\/        X  \\
   /\\ /    \\       / \\  \\
  /  X      \\_____/   \\  \\
 /  / \\                \\  \\
/  /   \\________________\\  \\
`,
]

const CHARS = ' .:-=+*#%@'

function scramble(text: string, progress: number): string {
  return text
    .split('')
    .map((ch) => {
      if (ch === ' ' || ch === '\n') return ch
      if (Math.random() < progress) return ch
      return CHARS[Math.floor(Math.random() * CHARS.length)]
    })
    .join('')
}

export default function Splash({ onDone }: { onDone: () => void }) {
  const [frameIndex, setFrameIndex] = useState(0)
  const [displayed, setDisplayed] = useState('')
  const [showTitle, setShowTitle] = useState(false)
  const [titleScale, setTitleScale] = useState(0.3)
  const [titleOpacity, setTitleOpacity] = useState(0)
  const [fadeOut, setFadeOut] = useState(false)

  const finalFrame = MOUNTAIN_FRAMES[MOUNTAIN_FRAMES.length - 1]

  const animateMountain = useCallback(() => {
    let frame = 0
    const interval = setInterval(() => {
      if (frame < MOUNTAIN_FRAMES.length) {
        setFrameIndex(frame)
        setDisplayed(scramble(MOUNTAIN_FRAMES[frame], 0.3))
        frame++
      } else {
        clearInterval(interval)
        let resolveProgress = 0
        const resolveInterval = setInterval(() => {
          resolveProgress += 0.15
          if (resolveProgress >= 1) {
            setDisplayed(finalFrame)
            clearInterval(resolveInterval)
            setShowTitle(true)
          } else {
            setDisplayed(scramble(finalFrame, resolveProgress))
          }
        }, 50)
      }
    }, 180)
    return interval
  }, [finalFrame])

  useEffect(() => {
    const interval = animateMountain()
    return () => clearInterval(interval)
  }, [animateMountain])

  useEffect(() => {
    if (!showTitle) return
    let raf: number
    const start = performance.now()
    const duration = 800

    function tick(now: number) {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setTitleScale(0.3 + eased * 0.7)
      setTitleOpacity(eased)
      if (t < 1) {
        raf = requestAnimationFrame(tick)
      } else {
        setTimeout(() => setFadeOut(true), 400)
        setTimeout(onDone, 900)
      }
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [showTitle, onDone])

  return (
    <div
      className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-black transition-opacity duration-500 ${fadeOut ? 'opacity-0' : 'opacity-100'}`}
    >
      <pre className="font-mono text-[10px] leading-[1.2] text-white/60 sm:text-xs">
        {displayed || MOUNTAIN_FRAMES[frameIndex]}
      </pre>

      {showTitle && (
        <div
          className="mt-6 font-mono text-4xl font-light tracking-[0.3em] text-white sm:text-5xl"
          style={{
            transform: `scale(${titleScale})`,
            opacity: titleOpacity,
          }}
        >
          clmbr
        </div>
      )}
    </div>
  )
}
