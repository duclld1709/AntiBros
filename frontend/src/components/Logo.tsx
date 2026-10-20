export function Logo({ size = 36 }: { size?: number }) {
  return (
    <svg
      className="logo-mark"
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-hidden="true"
      focusable="false"
    >
      <rect width="64" height="64" rx="16" className="logo-bg" />
      <g className="logo-bar">
        <rect x="8" y="22" width="8" height="20" rx="3" />
        <rect x="48" y="22" width="8" height="20" rx="3" />
        <rect x="16" y="18" width="7" height="28" rx="3" />
        <rect x="41" y="18" width="7" height="28" rx="3" />
        <rect x="23" y="29" width="18" height="6" rx="3" />
      </g>
    </svg>
  );
}
