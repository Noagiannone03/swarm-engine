# Plan d'implémentation — Porte de contribution (« tu contribues = tu consommes »)

> **Pour l'exécutant (IA ou dev)** : document autonome, ancres fichier:ligne
> vérifiées sur `fabi-patches` (`Noagiannone03/swarm-engine`) et sur le VPS de
> prod. Exécuter les phases dans l'ordre. Contraintes §6 non négociables.

---

## 0. Objectif produit & philosophie

Fabi : *« tu utilises = tu contribues »*. La règle est **binaire**, sans
comptabilité de points :

- **Ton compte a un worker actif dans le swarm → tu consommes autant que tu veux.**
- **Pas de worker actif → pas d'accès** (réponse claire qui explique comment
  contribuer ; le CLI/IDE enchaîne en lançant le worker automatiquement).

Pas de monnaie, pas de solde, pas de marché : une **porte**. C'est plus simple
que les kudos d'AI Horde et suffisant car notre scheduler est centralisé : il
**voit tout** (il assigne les couches, reçoit les heartbeats toutes les 10 s,
route chaque requête). Aucun client ne déclare quoi que ce soit → rien à
falsifier.

### Décision produit assumée
Une machine qui ne PEUT pas contribuer (ni Apple Silicon ni GPU NVIDIA) ne peut
pas consommer le swarm public. C'est la philosophie du réseau. Échappatoires :
`FABI_GATE=off` pour les déploiements privés/dev, et une allowlist d'admin.

---

## 1. Contexte vérifié

### 1.1 Côté code (`fabi-patches`)

| Point d'ancrage | Fichier | Rôle |
|---|---|---|
| API conso (LE point de passage unique) | `src/backend/main.py` **l.183** : `@app.post("/v1/chat/completions")` (FastAPI) | y poser la porte |
| Heartbeat reçu par le scheduler | `src/backend/server/rpc_connection_handler.py` **l.72** `node_update()` → l.95 `scheduler.enqueue_node_update(...)` | y rafraîchir le bail (lease) |
| Node info envoyé par le worker | `src/parallax/p2p/server.py` **l.949** `get_node_info()`, dict construit **l.985** (`"node_id": ...`) | y ajouter le token de compte |
| Args du worker | `src/parallax/launch.py` + `src/parallax/server/server_args.py` (+ relai des args inconnus par `parallax join`, `src/parallax/cli.py` `join_command`) | `--account-token` / env |
| Éviction des nœuds morts | timeout heartbeat 25 s (env `PARALLAX_HEARTBEAT_TIMEOUT`, cf. `.env` prod) + backoff fiabilité Petals (commit `debcd20`) | un faux/mauvais worker cesse de compter automatiquement |

### 1.2 Côté prod (VPS OVH 37.59.98.16, audité)

- **5 conteneurs scheduler** (un par modèle : 1.7B/8B/Coder-30B/Coder-480B/GLM-4.5),
  compose dans `~/parallax-scheduler`, `network_mode: host`, ports 3001/3011-3014.
- **Caddy** (`~/edge-proxy/Caddyfile`) : TLS sur `server.undefinedstudio.fr/fabi-scheduler/<modèle>`.
- **Redis déjà présent** : conteneur `vago-redis` (redis:7-alpine, healthy,
  127.0.0.1:6379) → le store partagé idéal pour les baux inter-swarms.
- ⚠️ Constats sécurité (à régler en Phase 4, prérequis d'efficacité de la
  porte) : ports 3001-3014 **ouverts au public sans auth**, pas de firewall ;
  images **périmées** (commit `749460e` du 12 mai, ~15 commits de retard).

---

## 2. Conception — le « bail de contribution » (lease)

```
   worker (machine de l'user)                    scheduler (VPS)
   ─────────────────────────                     ───────────────────────────
   parallax join --account-token T   ──────►     node_update (10 s) :
                                                 le nœud est-il dans la table
                                                 active du scheduler ?
                                                   oui → SETEX lease:H(T) 300s
                                                         {"node_id", "model", ts}
   client (CLI/IDE)                              (redis partagé entre les 5
   Authorization: Bearer T            ──────►     schedulers → contribuer à UN
   POST /v1/chat/completions                      swarm débloque TOUS les modèles)
                                                 EXISTS lease:H(T) ?
                                                   oui → servir (illimité)
                                                   non → 402 contribution_required
```

Points de conception (réfléchis, ne pas « simplifier ») :

1. **Le bail n'est rafraîchi QUE par le scheduler lui-même**, à partir de SA
   table de nœuds actifs (jamais sur déclaration du client). Un worker évincé
   (timeout 25 s, blacklist backoff) cesse d'être rafraîchi → la porte se
   referme toute seule à l'expiration du TTL. Zéro code de révocation.
2. **TTL = 300 s** (env `FABI_GATE_LEASE_S`). Pourquoi 5 min et pas 30 s :
   absorbe un redémarrage de worker, une réallocation de couches (reload du
   modèle) ou un blip réseau **sans couper un chat en cours**. Le heartbeat
   (10 s) rafraîchit ~30× par fenêtre → aucun flapping.
3. **Warmup couvert naturellement** : un worker qui télécharge/charge le modèle
   heartbeat déjà (status `INITIALIZING`) et figure dans la table du scheduler
   → il compte comme contributeur pendant le warmup. Première expérience
   utilisateur : « Se connecter » → accès immédiat.
4. **`H(T)` = SHA-256 du token** : redis ne stocke jamais le secret. Le token
   transite uniquement dans des canaux chiffrés (TLS via Caddy côté client ;
   canal lattica/libp2p côté worker).
5. **Inter-swarms via le redis partagé** : un Mac 16 Go qui héberge des couches
   de Qwen3-8B peut consommer Coder-480B. C'est voulu : « prête ce que tu peux,
   utilise ce dont tu as besoin ». **Fallback sans redis** : dict en mémoire du
   process (la porte devient par-swarm — dégradation acceptable, documentée).
6. **La porte ne s'applique qu'à l'admission** de la requête, jamais au milieu
   d'un stream (une génération entamée se termine toujours).
7. **Modes** (env `FABI_GATE`) : `off` (défaut upstream/dev — comportement
   actuel intact), `on` (notre prod). + `FABI_GATE_ALLOWLIST` (tokens admin
   séparés par virgule, ex. pour le monitoring).
8. **Réponse 402 normalisée** (consommée par le CLI/IDE pour l'UX) :
   ```json
   {"error": {"code": "contribution_required",
              "message": "Connecte un worker au swarm pour utiliser Fabi",
              "join_command": "parallax join -s <peer> --account-token <ton token>"}}
   ```

### Anti-bypass (analyse)
- **Appel direct de l'API sans contribuer** → 402. Les ports bruts 3001-3014
  sont fermés en Phase 4 → Caddy est l'unique chemin → porte incontournable.
- **Token volé/partagé** → c'est un bearer secret personnel ; le partager =
  partager son compte (acceptable à notre échelle ; les canaux sont chiffrés).
- **Faux worker** (rejoint puis ne sert rien) → il doit passer par la vraie
  table du scheduler : nœud qui échoue → évincé/blacklisté (mécanismes
  existants) → bail expiré. Pas de demi-mesure à coder.
- **Spam de connexions** : rate-limit Caddy sur `/fabi-scheduler/*` (Phase 4).

---

## 3. Phase 1 — scheduler (swarm-engine, `src/backend/`)

**Nouveau fichier `src/backend/server/contribution_gate.py`** (~120 lignes) :

```python
class ContributionGate:
    def __init__(self): ...           # lit FABI_GATE, FABI_GATE_LEASE_S,
                                      # FABI_GATE_REDIS_URL (defaut redis://127.0.0.1:6379/0),
                                      # FABI_GATE_ALLOWLIST ; redis optionnel →
                                      # fallback dict {hash: expiry} en mémoire
    def refresh(self, account_token: str, node_id: str, model: str) -> None
    def is_allowed(self, account_token: str | None) -> bool
    def denial_response(self) -> JSONResponse   # le 402 normalisé ci-dessus
```

- Import redis **gardé** (`try/except ImportError`) — pas de nouvelle dépendance
  dure (le paquet `redis` n'est ajouté qu'à l'image Docker du scheduler, pas à
  `pyproject.toml` core ; si absent → fallback mémoire).
- **Brancher le refresh** : dans `rpc_connection_handler.py::node_update`
  (l.72-95), après que le nœud est accepté/connu du scheduler : si le message
  contient `account_token`, appeler `gate.refresh(...)`. NE PAS rafraîchir pour
  un nœud inconnu/rejeté.
- **Brancher la porte** : dans `src/backend/main.py`, sur
  `POST /v1/chat/completions` (l.183) : lire `Authorization: Bearer` →
  `gate.is_allowed(token)` sinon retour `gate.denial_response()`. (Les autres
  endpoints — `/cluster/status_json`, UI — restent ouverts : ils ne consomment
  pas de calcul.)
- **Tests** (`tests/` nouveau `test_contribution_gate.py`) : gate off →
  passthrough ; on + bail valide → 200 ; on + token inconnu/absent → 402 avec
  le JSON exact ; expiration TTL → 402 ; allowlist → 200 ; fallback sans redis.

## 4. Phase 2 — worker (swarm-engine)

- `server_args.py` : `--account-token` (str, défaut env `FABI_ACCOUNT_TOKEN`,
  sinon None). `parallax join` relaie déjà les args inconnus vers `launch.py`.
- `p2p/server.py::get_node_info` (l.985) : ajouter
  `"account_token": self.account_token` si défini (le canal RPC lattica est
  chiffré). Plomber l'attribut depuis `launch.py`.
- Rien d'autre côté worker : il ne calcule rien, ne prouve rien.

## 5. Phase 3 — clients (repos `fabi-cli` + `fabi-ide`)

- **Génération du token** (une fois) : 32 octets aléatoires hex →
  `~/.config/fabi/account-token` (0600). Partagé par le CLI et l'IDE.
- **Conso** : le token devient l'`apiKey` OpenAI (l'IDE met aujourd'hui
  `'fabi-no-auth'` dans `fabi-swarm/src/browser/fabi-swarm-model.ts` — y mettre
  le token lu via le backend Theia ; pareil côté fabi-cli launcher).
- **Contribution** : `spawnWorker` (IDE : `fabi-swarm/src/node/fabi-swarm-worker.ts`,
  + `FabiRuntimeManager.joinArgs()`) et le launcher fabi-cli passent
  `--account-token <T>` au `parallax join`.
- **UX du 402** : intercepter `contribution_required` → message « Connecte-toi
  au swarm pour utiliser Fabi » + déclencher le flux « Se connecter » (worker
  auto), puis retry. (L'IDE a déjà le panneau + le bouton.)

## 6. Phase 4 — déploiement prod (VPS) & durcissement (prérequis d'efficacité)

1. **Rebuild des 5 images** (elles sont à `749460e`, 15 commits de retard) :
   `cd ~/parallax-scheduler && docker compose build --no-cache --pull && docker compose up -d`.
2. `.env` : `FABI_GATE=on`, `FABI_GATE_REDIS_URL=redis://127.0.0.1:6379/2`
   (réutilise `vago-redis`, host network → joignable), `FABI_GATE_LEASE_S=300`.
3. **Firewall** (sinon la porte est contournable par les ports bruts) : ufw
   allow 22, 443, 18080-18140/tcp+udp (Lattica P2P, nécessaires aux workers) ;
   **deny 3001-3014** depuis l'extérieur (Caddy y accède en local). Vérifier
   après coup que `curl http://37.59.98.16:3001/...` échoue depuis l'extérieur.
4. **Caddy** : rate-limit raisonnable sur `/fabi-scheduler/*` (anti-spam).
5. **Rollback** : `FABI_GATE=off` + `docker compose up -d` = retour instantané.

## 7. Contraintes non négociables

1. `FABI_GATE=off` par défaut dans le code (upstream-friendly) ; activé
   explicitement en prod. Comportement actuel strictement inchangé porte off.
2. Aucune nouvelle dépendance dure dans `pyproject.toml` (redis = optionnel
   gardé, installé dans l'image Docker scheduler uniquement).
3. Le bail n'est JAMAIS rafraîchi sur déclaration d'un client — uniquement
   depuis la table de nœuds actifs du scheduler.
4. Ne pas casser les endpoints non-conso (`/cluster/status_json`, UI, registry).
5. La porte ne coupe jamais un stream en cours.
6. Secrets : jamais le token en clair dans les logs ni dans redis (hash only).
7. Branche cible **`fabi-patches`** ; commits atomiques par phase, style `fabi:`.
8. E2E avant de déclarer fini : worker connecté → chat OK ; worker arrêté →
   après ≤ 300 s → 402 + UX de reconnexion ; `FABI_GATE=off` → tout passe.

## 8. Ordre d'exécution

| # | Tâche | Done quand |
|---|---|---|
| 1 | `contribution_gate.py` + hooks backend + tests | pytest vert, gate off = passthrough |
| 2 | `--account-token` worker → node_info | visible dans les logs scheduler (hashé) |
| 3 | Clients : token + apiKey + join arg + UX 402 | IDE : « Se connecter » → chat ; déconnecté → message clair |
| 4 | Prod : rebuild + redis + ufw + Caddy | curl direct port 3001 bloqué ; via Caddy sans token → 402 ; avec worker → 200 |
| 5 | Validation E2E complète (§7.8) | scénarios verts, documentés |
