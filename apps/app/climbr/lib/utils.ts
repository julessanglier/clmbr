
  export function getSlopeColor(slope: number) {
    const min = 0,
      max = 20;
    const t = Math.max(0, Math.min(1, (slope - min) / (max - min)));
    const r = Math.round(255 * t);
    const g = Math.round(255 * (1 - t));
    return `rgb(${r},${g},0)`;
  }