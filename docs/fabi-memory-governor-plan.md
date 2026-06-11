# Plan d'implémentation — Gouverneur mémoire dynamique (Fabi / Parallax)

> **Pour l'exécutant (IA ou dev)** : ce document est autonome. Tous les chemins,
> lignes et mécanismes cités ont été vérifiés sur la branche `fabi-patches` de ce
> repo (`Noagiannone03/swarm-engine`). Lis d'abord « Contexte vérifié », puis
> exécute les phases dans l'ordre. Ne saute pas les contraintes (§7).

---

## 0. Objectif produit

Un worker Fabi tourne sur le PC personnel de l'utilisateur (Mac Apple Silicon,
Linux NVIDIA, Windows NVIDIA) et **prête** de la mémoire au swarm. Si
l'utilisateur ouvre un gros programme (Photoshop, un jeu, une VM), le worker
doit **se serrer tout seul** — réduire son empreinte, jusqu'à rendre des couches
au swarm — **sans osciller** (pas de yo-yo de rechargements) et **sans jamais
geler la machine**. Quand la pression retombe, il regrandit **lentement**.

C'est le comportement d'un thermostat, pas d'un interrupteur.

---

## 1. Contexte vérifié (ne pas re-découvrir, c'est fait)

### 1.1 Ce qui existe déjà et MARCHE (ne pas réécrire)

| Mécanisme | Fichier (branche `fabi-patches`) | Détail |
|---|---|---|
| Budget mémoire étagé (hôtes à mémoire partagée, Mac) | `src/parallax/server/server_info.py` → `_resolve_usable_memory_gb()` | min(total−réserve_étagée, total×0.45, cap_pression_psutil, Metal_recommended×0.45), plancher 1 Go. Réserve étagée : 6→10 Go selon la taille. Env overrides `PARALLAX_*`. |
| Cap de pression instantané | `server_info.py` → `_memory_pressure_cap_gb()` | basé sur `psutil.virtual_memory().available`, multiplicateurs gradués ×0.35/0.55/0.75 selon le ratio dispo. |
| Plafond GPU officiel Apple | `server_info.py` → `_recommended_metal_memory_gb()` | `mx.metal.device_info()["max_recommended_working_set_size"]` — même signal que llama.cpp/Ollama. |
| Limite wired MLX | `server_info.py` → `resolve_mlx_wired_limit_bytes()` | budget×0.9, posée au démarrage. |
| Budget VRAM CUDA | `src/parallax_utils/cuda_memory.py` → `resolve_cuda_memory_budget(total_gb, free_gb)` | réserve + fraction utilisable + prise en compte de `free_gb` si fourni. |
| **Re-mesure périodique** | `src/parallax/p2p/server.py` → `_announcer_thread()` (~l.818) appelle `get_node_info(is_update=True)` toutes les **10 s** (heartbeat) ; `get_node_info` (l.949) appelle `detect_node_hardware()` (`server_info.py` l.472) qui relit `psutil.virtual_memory()` **en vif, sans cache** (vérifié : aucun lru_cache sur le chemin). |
| **Réallocation de couches à chaud** | `p2p/server.py` ~l.840-875 : si la réponse du heartbeat contient un autre `start_layer/end_layer/model_name` → `self._layer_allocation_changed = True`, statut → `INITIALIZING`, sync `shared_state` ; `src/parallax/server/executor/base_executor.py` l.485 : `run_loop` lit `shared_state.get_layer_allocation_changed()` → sort de la boucle → l'executor est rechargé avec les nouvelles couches. |
| Allocation pondérée mémoire côté scheduler central | `src/scheduling/layer_allocation.py` (classe `BaseLayerAllocator`, param **`rebalance_threshold: float = 0.25`** l.104) ; entrée scheduler : `src/backend/server/rpc_connection_handler.py` → `node_update()` (l.72) → `scheduler.enqueue_node_update(...)` (l.95). |
| Events structurés vers l'app | logs `[FABI] {"event": ...}` (déjà émis pour `peer_id`, `allocated` — voir `p2p/server.py`). |
| Tests existants | `tests/scheduler_tests/test_memory_limits.py`, `tests/scheduler_tests/test_layer_allocation.py`. |

### 1.2 Stack de détection — déjà la bonne, ne rien ajouter d'exotique

- **psutil** : standard absolu multi-OS pour la RAM. Déjà dépendance, déjà utilisé.
- **VRAM vive NVIDIA** : `torch.cuda.mem_get_info()` (libre/total **device-wide**,
  voit donc les jeux/autres apps) — torch est déjà là sur le chemin CUDA.
  `nvidia-ml-py` (NVML, ce qu'Ollama utilise en interne) est déjà installé avec
  vLLM : utilisable en fallback, import gardé.
- **Mac** : Metal `recommendedMaxWorkingSetSize` (déjà utilisé).
- **Linux PSI** : lire `/proc/pressure/memory` directement (≈5 lignes). ⚠️ Le
  paquet PyPI nommé `PSI` est un FAUX AMI (vieille lib 2008 sans rapport) — ne
  pas l'installer.
- **Aucune lib "gouverneur" n'existe** : Ollama/Petals/exo écrivent tous leur
  propre logique de contrôle. C'est ce qu'on fait ici (~150 lignes).

### 1.3 Le problème précis à résoudre

La mesure est vive mais **instantanée** : si la RAM dispo oscille, le
`memory_gb` annoncé oscille → le scheduler central peut réallouer en boucle →
rechargements de modèle en yo-yo (cf. bug connu Ollama #4151 « yo-yoing memory
pressure »). Il manque : **lissage + hystérésis + riposte graduée + re-mesure
VRAM vive côté CUDA**.

---

## 2. Architecture cible — « échelle de riposte graduée + hystérésis »

```
   capteurs (psutil / mem_get_info / Metal / PSI)      ← existant + Phase 3
        │  toutes les 10 s (heartbeat existant)
        ▼
   ┌────────────────────────────┐
   │  MemoryGovernor (NOUVEAU)  │  lissage EMA + hystérésis + quantification
   │  src/parallax_utils/       │  → budget_annoncé (Go), palier de pression
   │  memory_governor.py        │     {NORMAL, ELEVATED, CRITICAL}
   └────────────────────────────┘
        │
        ├── Palier 0 (gratuit, immédiat)  : clamp admission (batch/tokens)     — pas de reload
        ├── Palier 1 (lourd, amorti)      : memory_gb réduit dans le heartbeat → scheduler
        │                                   central réalloue moins de couches → reload (existant)
        └── Palier 2 (urgence)            : statut INITIALIZING/pause, refus d'admission,
                                            shrink immédiat (bypass du dwell)
```

Principes :
- **Réagir vite à la pression, regrandir lentement** (asymétrie volontaire).
- **Le moins cher d'abord** : clamp d'admission (aucun reload) avant de rendre
  des couches (reload).
- **Quantifier** le budget annoncé (pas de variations < 1 Go ou < 10 %) pour que
  le scheduler ne voie jamais de bruit.

---

## 3. Phase 1 — le module `MemoryGovernor` (pur, testable, sans I/O)

**Nouveau fichier : `src/parallax_utils/memory_governor.py`**

Classe sans dépendance (pas de psutil dedans — on lui INJECTE les mesures, pour
la testabilité) :

```python
class MemoryGovernor:
    def __init__(self, *, now_fn=time.monotonic, **overrides): ...
    def observe(self, raw_budget_gb: float, *,
                available_ratio: float | None = None,   # dispo/total système
                psi_avg10: float | None = None           # Linux PSI, sinon None
                ) -> GovernorDecision: ...
```

`GovernorDecision` (dataclass) : `advertised_gb: float`,
`pressure: Literal["NORMAL","ELEVATED","CRITICAL"]`,
`admission_scale: float` (1.0 = pas de clamp, 0.5 = moitié du batch, etc.),
`changed: bool` (True seulement si `advertised_gb` a changé après quantification).

Logique interne (tous les seuils = constantes module surchargeables par env
`PARALLAX_GOV_*`, défauts ci-dessous) :

1. **EMA** du `raw_budget_gb` : `alpha = 0.3` (≈ converge en ~5 ticks de 10 s).
2. **Palier de pression** :
   - `CRITICAL` si `available_ratio < 0.08` OU `psi_avg10 > 25`.
   - `ELEVATED` si `available_ratio < 0.18` OU `psi_avg10 > 10` OU
     `ema < advertised × 0.85` .
   - sinon `NORMAL`.
3. **Hystérésis sur le budget annoncé** :
   - **Shrink** : si `ema < advertised × (1 − 0.15)` pendant **3 ticks
     consécutifs** (30 s) → nouveau `advertised = quantize(ema)`.
   - **Grow** : si `ema > advertised × (1 + 0.25)` pendant **30 ticks
     consécutifs** (5 min) → `advertised = quantize(min(ema, advertised × 1.25))`
     (croissance par paliers de 25 % max — jamais d'un coup).
   - **Dwell** : minimum **10 min** (`PARALLAX_GOV_DWELL_S=600`) entre deux
     changements de `advertised` — SAUF si `CRITICAL` (bypass immédiat, shrink
     direct à `quantize(ema × 0.7)`).
4. **Quantification** : `quantize()` arrondit au pas de `max(1.0 Go, 10 % de la
   valeur)` — garantit qu'un petit bruit ne produit jamais `changed=True`.
5. **admission_scale** (Palier 0, indépendant du dwell, réagit en 1 tick) :
   `NORMAL → 1.0`, `ELEVATED → 0.5`, `CRITICAL → 0.0` (= ne plus admettre de
   nouvelles requêtes ; celles en cours se terminent).
6. **Kill-switch** : si env `PARALLAX_MEMORY_GOVERNOR=0` → `observe()` retourne
   le raw passthrough (comportement actuel). Défaut : activé.

**Tests unitaires (nouveau `tests/scheduler_tests/test_memory_governor.py`)** —
utiliser `now_fn` injecté pour simuler le temps :
- entrée stable → `changed` jamais True après convergence ;
- oscillation ±20 % autour d'une moyenne → **zéro** changement (anti-yo-yo) ;
- chute brutale soutenue → shrink après exactement 3 ticks ;
- chute + remontée < 3 ticks → aucun changement ;
- `CRITICAL` → shrink immédiat même pendant le dwell ;
- remontée → premier grow seulement après 5 min, par palier ≤ 25 % ;
- dwell de 10 min respecté entre deux changements non critiques ;
- quantification : 12.3→12.0, variations < pas → `changed=False` ;
- env overrides pris en compte ; kill-switch = passthrough.

---

## 4. Phase 2 — branchement worker (capteurs → gouverneur → effets)

### 4.1 Brancher le gouverneur dans la mesure heartbeat

**Fichier : `src/parallax/server/server_info.py`**, fonction
`detect_node_hardware()` (l.472).

- Instancier UN `MemoryGovernor` global module (singleton paresseux — le
  heartbeat est le seul appelant périodique, thread unique `_announcer_thread`).
- Chemin mémoire partagée (Mac, et CPU) : aujourd'hui ça retourne
  `_resolve_usable_memory_gb(...)` direct → passer cette valeur dans
  `governor.observe(raw, available_ratio=available/total, psi_avg10=...)` et
  retourner `decision.advertised_gb`.
- Chemin CUDA : voir 4.2 (re-mesure VRAM vive d'abord), puis même passage par
  le gouverneur.
- Ajouter au dict hardware retourné : `"pressure": decision.pressure` (champ
  additionnel inoffensif pour le scheduler, utile au debug/UI).

### 4.2 VRAM vive côté CUDA (Linux ET Windows — même chemin, déjà validé)

Dans le chemin CUDA de `detect_node_hardware()` :
- `free_b, total_b = torch.cuda.mem_get_info()` (device-wide : voit les jeux et
  autres process) — import déjà gardé dans ce fichier.
- Fallback si exception : `pynvml` (`nvidia-ml-py`, présent avec vLLM), import
  dans un try/except ; sinon `free_gb=None` (comportement actuel).
- Passer `free_gb` à `resolve_cuda_memory_budget(total_gb, free_gb)` — la
  signature l'accepte déjà (`src/parallax_utils/cuda_memory.py` l.54-55).

### 4.3 Linux PSI (bonus, ~10 lignes, NE PAS installer de lib)

Helper dans `memory_governor.py` (seule I/O tolérée, gardée) :

```python
def read_psi_memory_avg10() -> float | None:
    try:
        with open("/proc/pressure/memory") as f:   # absent hors Linux/anciens kernels
            line = f.readline()                     # "some avg10=0.00 avg60=..."
        return float(line.split("avg10=")[1].split()[0])
    except Exception:
        return None
```

### 4.4 Palier 0 — clamp d'admission en vif (sans reload)

- **`src/parallax/utils/shared_state.py`** : ajouter get/set
  `admission_scale` (float, défaut 1.0) — suivre le modèle des champs existants
  (`get_layer_allocation_changed` l.124, `update_metrics` l.81).
- **`src/parallax/p2p/server.py`** (`_announcer_thread`) : après chaque
  `governor.observe(...)` (via le hardware info), écrire
  `shared_state.set("admission_scale", decision.admission_scale)` et émettre un
  event structuré :
  `[FABI] {"event":"pressure","level":...,"advertised_gb":...,"admission_scale":...}`
  (suivre le format des events existants).
- **`src/parallax/server/scheduler.py`** (le scheduler LOCAL de l'executor,
  classe `Scheduler`) : dans `admit_requests()` (et/ou `form_batch()`), lire
  `shared_state.get("admission_scale", 1.0)` (le `Scheduler` reçoit déjà
  `shared_state`, cf. `base_executor.py` l.161-173) et clamper :
  `effective_max_batch = max(1, int(max_batch_size * scale))` si scale > 0 ;
  si `scale == 0.0` → ne plus admettre (les requêtes en file restent, celles en
  cours finissent). NE PAS toucher au KV pool vLLM à chaud (impossible
  post-init) — le clamp d'admission suffit comme palier 0.

### 4.5 Palier 1 — il est automatique

Rien à coder de plus : le `advertised_gb` réduit part dans le heartbeat
(`memory_gb`), le scheduler central réalloue (mécanisme §1.1), l'executor se
recharge. **Vérifier** seulement (§5) que le scheduler central réagit bien à un
changement de `memory_gb` d'un nœud existant.

### 4.6 Palier 2 — urgence

Si `decision.pressure == "CRITICAL"` : en plus du shrink immédiat (géré par le
gouverneur) et de `admission_scale=0`, log warning clair. NE Pas tuer le
process : l'objectif est de dégonfler, le reload de couches suit au heartbeat
suivant.

---

## 5. Phase 3 — côté scheduler central (vérifier + amortir)

**Fichiers : `src/backend/server/rpc_connection_handler.py` (l.72-113),
`src/scheduling/layer_allocation.py`, `src/scheduling/scheduler.py` (ou
équivalent — suivre `enqueue_node_update`).**

1. **Tracer le chemin** `enqueue_node_update` → où le `memory_gb` mis à jour
   est-il appliqué au `Node`, et qu'est-ce qui déclenche une réallocation ?
   (Le `BaseLayerAllocator` a déjà `rebalance_threshold=0.25`.)
2. **Garantir** : une baisse de `memory_gb` > 20 % d'un nœud déclenche une
   réallocation ; une variation < 10 % n'en déclenche JAMAIS (le gouverneur
   quantifie déjà, ceinture-bretelles côté scheduler).
3. **Amortisseur scheduler** : ne pas réallouer le pipeline global plus d'une
   fois toutes les `SCHED_REALLOC_MIN_INTERVAL_S` (défaut 300 s) **sauf** si un
   nœud part/rejoint ou passe CRITICAL. Si un mécanisme équivalent existe déjà
   (todo l.479 `scheduler_manage.py` « rebalance status »), le compléter plutôt
   qu'en créer un deuxième.
4. **Test** (`tests/scheduler_tests/`) : nœud dont la mémoire chute de 40 % →
   nouvelle allocation avec moins de couches pour lui ; chute de 5 % → aucune
   réallocation ; deux chutes rapprochées → une seule réallocation dans la
   fenêtre.

---

## 6. Phase 4 — validation de bout en bout (manuelle, documenter les résultats)

1. **Mac (réel)** : lancer un worker (`parallax join` sur le swarm de test),
   puis ouvrir une grosse charge mémoire (ex.
   `python -c "b=[bytearray(1024**3) for _ in range(N)]; input()"`).
   Attendre 30-60 s. Vérifier dans les logs, dans l'ordre :
   `[FABI] pressure ELEVATED` → `admission_scale` réduit → si la pression
   persiste : `advertised_gb` réduit UNE fois → réallocation → reload. Libérer
   la mémoire → AUCUN grow avant 5 min, puis paliers ≤ 25 %. **Aucun yo-yo.**
2. **GPU CUDA (pod RunPod, cf. mémoire projet)** : worker actif, puis dans un
   autre process allouer ~30 % de la VRAM
   (`python -c "import torch; x=torch.empty(int(12e9//2), dtype=torch.float16, device='cuda'); input()"`).
   Vérifier que `free_gb` chute au heartbeat suivant et que la même séquence
   graduée se déroule. (Windows = même chemin de code, validé mlx-free.)
3. **Anti-régression** : machine au repos → le budget annoncé doit être
   identique à l'actuel (mêmes formules de base, gouverneur transparent à
   l'équilibre) ; `PARALLAX_MEMORY_GOVERNOR=0` → comportement strictement
   actuel.

---

## 7. Contraintes non négociables

1. **Ne PAS changer les formules de budget existantes** (réserves étagées,
   fraction 0.45, caps) — elles sont éprouvées sur le terrain. Le gouverneur
   s'ajoute PAR-DESSUS la mesure, il ne la remplace pas.
2. **Aucune nouvelle dépendance dure.** psutil existe ; pynvml = optionnel
   gardé ; PSI = lecture de fichier. Le paquet PyPI `PSI` est interdit (faux ami).
3. **Discipline d'imports multi-OS** : tout import mlx/torch/pynvml dans du code
   partagé doit être gardé `try/except ImportError` — même rigueur que le
   découplage mlx déjà fait (commit `fb60b28`). Le chemin Windows ne doit
   jamais importer mlx.
4. **Ne pas toucher la limite wired MLX à chaud** (posée au démarrage
   uniquement) ; ne pas tenter de rétrécir le KV pool vLLM à chaud (impossible
   post-init). Les leviers vifs sont : admission (palier 0) et couches (palier 1).
5. **Coût heartbeat négligeable** : tout le tick (psutil + mem_get_info + PSI +
   gouverneur) doit rester < 5 ms. Pas de subprocess dans le tick.
6. **Mac/Linux par défaut inchangés à l'équilibre** : sur une machine au repos
   le comportement observable doit être identique à aujourd'hui.
7. **Logs structurés** : chaque transition (palier, advertised_gb, raison) émet
   UN event `[FABI] {...}` — c'est ce que l'IDE/CLI affichent à l'utilisateur.
8. **Tout est surchargeable par env** (`PARALLAX_GOV_*`), avec les défauts du
   §3 ; documenter chaque variable dans `docs/user_guide/install.md` (section
   existante des budgets mémoire).
9. **Branche cible : `fabi-patches`** (c'est elle que la CI du repo `fabi`
   build, PAS `main`). Commits atomiques par phase, messages en anglais,
   style des commits `fabi:` existants.

---

## 8. Ordre d'exécution & definition of done

| # | Tâche | Done quand |
|---|---|---|
| 1 | `memory_governor.py` + tests unitaires | tous les tests §3 verts, `pytest tests/scheduler_tests/test_memory_governor.py` |
| 2 | Branchement `detect_node_hardware` + VRAM vive + PSI | budget annoncé stable au repos ; `pressure` dans node info ; tests existants `test_memory_limits.py` toujours verts |
| 3 | Palier 0 (admission_scale via shared_state → Scheduler.admit_requests) | test unitaire : scale 0.5 → batch effectif réduit ; scale 0 → aucune admission |
| 4 | Vérif/amortisseur scheduler central | tests §5 verts |
| 5 | Validation manuelle §6 (Mac + pod GPU) | séquence graduée observée, zéro yo-yo, logs `[FABI] pressure` corrects |
| 6 | Doc env vars + push `fabi-patches` | CI repo `fabi` toujours verte (tag rc) |
