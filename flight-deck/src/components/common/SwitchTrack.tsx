// The visual half of an on/off switch: a pill track with a sliding knob.
// Decorative — the enclosing <button role="switch"> carries the state and the
// name. The knob is anchored with left-0.5: an absolutely positioned child with
// no inset sits at its static position, which inside a <button> is the centre
// of the button, so the "on" knob used to slide off the track.

export function SwitchTrack({ on }: { on: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`relative inline-block h-4 w-7 shrink-0 rounded-full transition-colors ${
        on ? 'bg-sky-500' : 'bg-zinc-600'}`}
    >
      <span className={`absolute left-0.5 top-0.5 h-3 w-3 rounded-full bg-white shadow-sm transition-transform ${
        on ? 'translate-x-3' : 'translate-x-0'}`} />
    </span>
  )
}
