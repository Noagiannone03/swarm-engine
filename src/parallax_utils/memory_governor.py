"""Memory governor — empêche le budget mémoire annoncé d'osciller (anti yo-yo).

Le worker re-mesure sa mémoire à chaque heartbeat (~10 s). La mesure est vive
mais instantanée : si la RAM dispo oscille, le `memory_gb` annoncé oscillerait,
et le scheduler central réallouerait les couches en boucle → rechargements de
modèle en yo-yo (cf. bug connu Ollama #4151). Ce module pose un **gouverneur**
par-dessus la mesure :

- lisse la mesure brute (EMA),
- applique une **hystérésis asymétrique** : on RÉTRÉCIT vite sous pression
  soutenue, on REGRANDIT lentement et par paliers ;
- impose un **délai minimum** entre deux changements (sauf urgence) ;
- ne signale un changement que s'il dépasse un **pas minimum** (anti-bruit) ;
- expose un **palier de riposte gratuit** (`admission_scale`) qui réagit en 1
  tick sans rien recharger (le scheduler local réduit le batch admis).

Module PUR : aucune I/O dans la logique (la mesure lui est INJECTÉE), `now_fn`
injectable → tests déterministes. Seul `read_psi_memory_avg10()` lit un fichier
(Linux), gardé. Kill-switch : `PARALLAX_MEMORY_GOVERNOR=0` → passthrough.

Tous les seuils sont surchargeables par env `PARALLAX_GOV_*`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)

_FALSY = {"0", "false", "no", "off"}


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass
class GovernorDecision:
    advertised_gb: float           # budget à annoncer au scheduler
    pressure: str                  # "NORMAL" | "ELEVATED" | "CRITICAL"
    admission_scale: float         # 1.0 = plein ; 0.5 = moitié ; 0.0 = stop admission
    changed: bool                  # True seulement si advertised_gb a changé ce tick


class MemoryGovernor:
    def __init__(self, *, now_fn: Callable[[], float] = time.monotonic) -> None:
        self._now = now_fn
        self.enabled = os.environ.get("PARALLAX_MEMORY_GOVERNOR", "1").strip().lower() not in _FALSY

        # EMA + hystérésis
        self.alpha = _envf("PARALLAX_GOV_EMA_ALPHA", 0.3)
        self.shrink_ratio = _envf("PARALLAX_GOV_SHRINK_RATIO", 0.15)   # ema < adv*(1-r) → candidat shrink
        self.grow_ratio = _envf("PARALLAX_GOV_GROW_RATIO", 0.25)       # ema > adv*(1+r) → candidat grow
        self.shrink_ticks = _envi("PARALLAX_GOV_SHRINK_TICKS", 3)      # 3 ticks ≈ 30 s
        self.grow_ticks = _envi("PARALLAX_GOV_GROW_TICKS", 30)         # 30 ticks ≈ 5 min
        self.grow_step = _envf("PARALLAX_GOV_GROW_STEP", 0.25)         # +25 % max par palier
        self.dwell_s = _envf("PARALLAX_GOV_DWELL_S", 600.0)            # 10 min entre changements (sauf CRITICAL)
        self.critical_shrink = _envf("PARALLAX_GOV_CRITICAL_SHRINK", 0.7)

        # Pas minimum pour signaler un changement (anti-bruit)
        self.quant_min_gb = _envf("PARALLAX_GOV_QUANT_MIN_GB", 1.0)
        self.quant_frac = _envf("PARALLAX_GOV_QUANT_FRAC", 0.10)

        # Seuils de pression
        self.crit_avail = _envf("PARALLAX_GOV_CRITICAL_AVAIL", 0.08)
        self.elev_avail = _envf("PARALLAX_GOV_ELEVATED_AVAIL", 0.18)
        self.crit_psi = _envf("PARALLAX_GOV_CRITICAL_PSI", 25.0)
        self.elev_psi = _envf("PARALLAX_GOV_ELEVATED_PSI", 10.0)
        self.floor_gb = _envf("PARALLAX_GOV_FLOOR_GB", 1.0)

        # État
        self._ema: Optional[float] = None
        self._advertised: Optional[float] = None
        self._last_change: float = self._now()
        self._shrink_streak = 0
        self._grow_streak = 0

    # ----- helpers ---------------------------------------------------------

    def _pressure(self, available_ratio: Optional[float], psi_avg10: Optional[float]) -> str:
        if (available_ratio is not None and available_ratio < self.crit_avail) or (
            psi_avg10 is not None and psi_avg10 > self.crit_psi
        ):
            return "CRITICAL"
        # ELEVATED : peu de RAM dispo, PSI élevé, OU la tendance lissée plonge
        trend_drop = (
            self._ema is not None
            and self._advertised is not None
            and self._ema < self._advertised * (1.0 - self.shrink_ratio)
        )
        if (
            (available_ratio is not None and available_ratio < self.elev_avail)
            or (psi_avg10 is not None and psi_avg10 > self.elev_psi)
            or trend_drop
        ):
            return "ELEVATED"
        return "NORMAL"

    @staticmethod
    def _admission_scale(pressure: str) -> float:
        return {"NORMAL": 1.0, "ELEVATED": 0.5, "CRITICAL": 0.0}[pressure]

    def _min_delta(self) -> float:
        adv = self._advertised or 0.0
        return max(self.quant_min_gb, adv * self.quant_frac)

    def _apply(self, candidate: float) -> bool:
        """Applique une nouvelle valeur si l'écart dépasse le pas minimum."""
        candidate = max(self.floor_gb, round(candidate, 2))
        if self._advertised is None or abs(candidate - self._advertised) >= self._min_delta():
            self._advertised = candidate
            self._last_change = self._now()
            self._shrink_streak = 0
            self._grow_streak = 0
            return True
        return False

    # ----- API -------------------------------------------------------------

    def observe(
        self,
        raw_budget_gb: float,
        *,
        available_ratio: Optional[float] = None,
        psi_avg10: Optional[float] = None,
    ) -> GovernorDecision:
        raw_budget_gb = max(self.floor_gb, float(raw_budget_gb))

        if not self.enabled:
            return GovernorDecision(round(raw_budget_gb, 2), "NORMAL", 1.0, False)

        # EMA
        if self._ema is None:
            self._ema = raw_budget_gb
        else:
            self._ema = self.alpha * raw_budget_gb + (1.0 - self.alpha) * self._ema

        # Premier passage : on annonce directement la mesure (pas d'hystérésis encore)
        if self._advertised is None:
            self._advertised = max(self.floor_gb, round(raw_budget_gb, 2))
            self._last_change = self._now()
            pressure = self._pressure(available_ratio, psi_avg10)
            return GovernorDecision(self._advertised, pressure, self._admission_scale(pressure), True)

        pressure = self._pressure(available_ratio, psi_avg10)
        changed = False
        ema = self._ema

        # Compteurs de tendance
        if ema < self._advertised * (1.0 - self.shrink_ratio):
            self._shrink_streak += 1
        else:
            self._shrink_streak = 0
        if ema > self._advertised * (1.0 + self.grow_ratio):
            self._grow_streak += 1
        else:
            self._grow_streak = 0

        dwell_ok = (self._now() - self._last_change) >= self.dwell_s

        if pressure == "CRITICAL":
            # Urgence : shrink immédiat, on ignore le dwell.
            changed = self._apply(ema * self.critical_shrink)
        elif self._shrink_streak >= self.shrink_ticks and dwell_ok:
            # Pression soutenue : rétrécir vers la tendance lissée.
            changed = self._apply(ema)
        elif self._grow_streak >= self.grow_ticks and dwell_ok:
            # Calme prolongé : regrandir LENTEMENT, par palier borné.
            changed = self._apply(min(ema, self._advertised * (1.0 + self.grow_step)))

        if changed:
            logger.info(
                "MemoryGovernor: advertised=%.2fGB pressure=%s (ema=%.2f raw=%.2f)",
                self._advertised,
                pressure,
                ema,
                raw_budget_gb,
            )

        return GovernorDecision(
            self._advertised, pressure, self._admission_scale(pressure), changed
        )

    def state(self) -> Dict[str, float]:
        return {
            "advertised_gb": self._advertised or 0.0,
            "ema_gb": self._ema or 0.0,
            "shrink_streak": self._shrink_streak,
            "grow_streak": self._grow_streak,
        }


_governor_singleton: Optional[MemoryGovernor] = None


def get_governor() -> MemoryGovernor:
    """Process-wide singleton — l'état (EMA, streaks, dwell) DOIT persister entre
    les heartbeats, donc on réutilise la même instance."""
    global _governor_singleton
    if _governor_singleton is None:
        _governor_singleton = MemoryGovernor()
    return _governor_singleton


def read_psi_memory_avg10() -> Optional[float]:
    """Linux Pressure Stall Information (avg10 sur la ligne `some`).

    ⚠️ NE PAS installer le paquet PyPI nommé `PSI` (faux ami, lib 2008 sans
    rapport). On lit directement le fichier kernel. None hors Linux / kernel
    ancien / fichier absent.
    """
    try:
        with open("/proc/pressure/memory", "r") as fid:
            first = fid.readline()  # "some avg10=0.00 avg60=0.00 avg300=0.00 total=..."
        return float(first.split("avg10=")[1].split()[0])
    except Exception:
        return None
