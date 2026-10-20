import { Moon, Sun } from "lucide-react";

import type { Copy, Lang } from "../i18n";
import { Logo } from "./Logo";

interface Props {
  t: Copy;
  lang: Lang;
  onLang: (lang: Lang) => void;
  theme: "dark" | "light";
  onTheme: () => void;
  healthy: boolean | null;
  onHome: () => void;
}

export function Header({ t, lang, onLang, theme, onTheme, healthy, onHome }: Props) {
  const health = healthy === null ? "pending" : healthy ? "up" : "down";
  return (
    <header className="topbar">
      <button className="brand" onClick={onHome} aria-label="AntiBros">
        <Logo />
        <span className="brand-text">
          <span className="brand-name">
            Anti<span className="accent">Bros</span>
          </span>
          <span className="brand-tagline">{t.tagline}</span>
        </span>
      </button>
      <div className="topbar-actions">
        <span className={`health health-${health}`} role="status">
          <span className="health-dot" aria-hidden="true" />
          <span className="health-label">{healthy === false ? t.offline : t.online}</span>
        </span>
        <div className="segmented" role="group" aria-label={t.language}>
          {(["vi", "en"] as const).map((code) => (
            <button
              key={code}
              className={code === lang ? "active" : ""}
              aria-pressed={code === lang}
              onClick={() => onLang(code)}
            >
              {code.toUpperCase()}
            </button>
          ))}
        </div>
        <button className="icon-button" onClick={onTheme} aria-label={t.theme} title={t.theme}>
          {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  );
}
