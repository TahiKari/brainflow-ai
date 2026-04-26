# TODO — Intégration Codestral

## Tâche 1 — Clé API + `call_codestral`
- [x] `router/config_routes.py` : `CODESTRAL_API_KEY` ajouté à `SUPPORTED_KEYS` + modèle `ApiKeysUpdate`.
- [x] `services/config_service.py` : `CODESTRAL_API_KEY` ajouté à `SUPPORTED_KEYS`.
- [x] `ui/config.html` : champ `CODESTRAL_API_KEY` (formulaire + JS).
- [x] `services/llm_service.py` : `call_codestral(prompt)` + chargement `CODESTRAL_API_KEY`.
- [x] Vérification lecture/écriture via `/api/config/keys`.

## Tâche 2 — Routage intelligent Codestral
- [x] `router/router.py` : `CODESTRAL_TRIGGER_KEYWORDS` + `_should_use_codestral()`.
- [x] `router/router.py` : chaîne dynamique (codestral en tête si règle vraie, sinon jamais appelé).
- [x] Préservation de la chaîne fallback existante.

## Tests
- [x] `test_codestral_routing.py` — 4/4 ✅
- [x] `test_mistral_routing.py` — 7/7 ✅ (non-régression)
- [x] `test_codestral_advanced.py` — 6/6 ✅
  - [x] Edge cases détecteur (accents, markdown, emojis, sous-chaînes, types invalides)
  - [x] `call_codestral` fallback `MISTRAL_API_KEY` quand `CODESTRAL_API_KEY` absente
  - [x] `call_codestral` erreur propre quand aucune clé disponible
  - [x] Concurrence : 6 requêtes en parallèle (3 code → codestral, 3 chat → gemini)
  - [x] Non-régression `code_agent.handle_code_task`
  - [x] End-to-end HTTP `/api/chat` → workflow → router → codestral

## Tâche 3 — Logs légers dans routing et llm_service
- [x] `services/llm_service.py` : logger `brainflow.llm`
  - [x] Provider + modèle + `duration_ms` sur succès (INFO)
  - [x] Erreurs HTTP / réseau / JSON / inattendues (WARNING) avec `duration_ms`
  - [x] Cache hit (DEBUG)
  - [x] Couverture : Groq, Mistral, Codestral, DeepSeek, OpenRouter (via `_post_openai_compatible`) + Gemini (branche dédiée)
- [x] `router/router.py` : logger `brainflow.router`
  - [x] Entrée : `task_type`, activation Codestral, chaîne de providers
  - [x] Succès : provider utilisé, `fallback_count`, `attempt_ms`, `total_ms`
  - [x] Échec par provider : provider + `attempt_ms` + erreur tronquée (WARNING)
  - [x] Échec total : liste des providers essayés + `total_ms` (ERROR)
- [x] Impact perf : `time.perf_counter()` (qqs ns) + I/O logs synchrone (<0.5 ms), négligeable.
- [x] Tests existants : 17/17 toujours verts ✅
- [x] `test_logs.py` — 8/8 ✅ (dédiés aux logs)
  - [x] Router : succès + provider + `total_ms`
  - [x] Router : fallback avec `attempt_ms` + erreur tronquée
  - [x] Router : échec total → ERROR `all_failed` + `providers_tried`
  - [x] Router : log d'entrée `codestral=True` pour `task_type=='code'`
  - [x] LLM : succès `provider=... model=... status=ok duration_ms=...`
  - [x] LLM : erreur HTTP avec `status=500` + `duration_ms`
  - [x] LLM : erreur réseau `network_error=...` + `duration_ms`
  - [x] LLM : Gemini (branche dédiée) `provider=gemini duration_ms=...`

## Tâche 4 — Robustesse planner (fallback Gemini → Mistral → Groq)
- [x] `agents/planner.py` : chaîne `_PLANNER_PROVIDERS = (gemini, mistral_medium, groq)` résolue dynamiquement via `getattr` (testable via `patch.object`).
- [x] `_call_planner_llm()` : try/except autour de chaque appel, aucune exception ne remonte.
- [x] `_is_quota_error()` : détecte 429 / quota / rate limit / too many requests (case-insensitive).
- [x] Sur quota Gemini → log INFO `quota_exceeded → fallback`, sur autre erreur → log WARNING.
- [x] Succès après fallback → log INFO `after_fallback_count=N`.
- [x] Tous en échec → log ERROR `planner all_failed` + dict `ok=False` (jamais d'exception).
- [x] Champ `provider` ajouté au dict retourné (traçabilité).
- [x] **Aucune autre partie du projet modifiée** (contrainte respectée).
- [x] `test_planner_fallback.py` — 16/16 ✅
  - [x] Détecteur de quota (7 sous-checks)
  - [x] Gemini OK → pas de fallback
  - [x] Gemini 429 → Mistral (moyen)
  - [x] Gemini 429 + Mistral 500 → Groq
  - [x] Tous en échec → dict d'erreur, pas d'exception
  - [x] Exception brute `RuntimeError` capturée
  - [x] Erreur non-quota déclenche aussi le fallback
  - [x] Input vide rejeté sans appel LLM
  - [x] Logs : quota → INFO, all_failed → ERROR
- [x] `test_planner_e2e.py` — 2/2 ✅ (thorough HTTP via ASGITransport)
  - [x] POST `/api/chat` : Gemini 429 planner + Gemini 429 router → Mistral (moyen) reprend toute la chaîne
  - [x] POST `/api/chat` : tous les providers planner en échec → `ok=False` + message d'erreur clair (pas d'exception)
- [x] **Live server + curl** (thorough, vraie API) ✅
  - [x] `GEMINI_API_KEY` remplacée à chaud par une clé invalide → `POST /api/chat` renvoie `ok=True`, plan en 3 sous-tâches, 3/3 succeeded via Mistral
  - [x] Clé Gemini restaurée ensuite (round-trip POST/GET `/api/config/keys`)

## Tâche 5 — Détection d'intention dans le workflow
- [x] `services/workflow.py` : ajout de `detect_intent(message)` (règles simples par mots-clés).
  - [x] mots-clés "direct"   : `répond`, `écris`, `génère`, `fais`
  - [x] mots-clés "analysis" : `explique`, `analyse`, `pourquoi`, `comment`
  - [x] défaut : `"direct"`
  - [x] comparaison case-insensitive, priorité "direct" si les deux familles cohabitent
  - [x] robustesse aux entrées non-str (None, int, dict) → `"direct"`
- [x] `run_workflow()` : calcul d'`intent_type` en amont du planner, log INFO dédié.
- [x] `intent_type` ajouté au dict retourné dans **tous les chemins** (succès, échec planner, sous-tâches vides).
- [x] **Aucune autre logique du workflow modifiée** (contrainte respectée).
- [x] `test_intent_detection.py` — 20/20 ✅
  - [x] 16 tests unitaires sur `detect_intent` (tous mots-clés, casse, sous-chaînes, ponctuation, priorité, défaut, types invalides)
  - [x] 4 tests d'intégration `run_workflow` : présence d'`intent_type` sur échec planner, sous-tâches vides, succès complet, non-régression champs historiques
- [x] Non-régression globale : 5 suites existantes re-vérifiées (codestral routing/advanced, mistral routing, logs, planner fallback) — toutes vertes ✅

## Tâche 6 — Branchement par `intent_type` dans `run_workflow`
- [x] `services/workflow.py` : nouvel helper `_run_direct(user_input)` — un seul appel LLM via `route_request("simple", user_input)`.
- [x] `run_workflow` : court-circuit immédiat vers `_run_direct` si `intent_type == "direct"` (avant planner).
- [x] Mode `direct` → dict simplifié : `{ok, intent_type, user_input, response, plan=[], steps=[], summary={total:0,...}}`.
- [x] Mode `analysis` → pipeline complet strictement **inchangé** (planner + dispatch + validator).
- [x] `router/chat_routes.py` : préfère `result["response"]` si présent (mode direct), sinon `_build_response_text(result)` (mode analysis) — fallback unique, minimalement invasif.
- [x] `test_intent_workflow_branch.py` — 8/8 ✅
  - [x] Direct : un seul `route_request`, planner **non** appelé, `response` peuplé
  - [x] Direct : échec LLM → `ok=False`, message d'erreur propagé, planner non appelé
  - [x] Analysis : planner + validator appelés, `plan`/`steps`/`summary` peuplés, pas de champ `response` en top-level
  - [x] Non-régression : tous les champs historiques préservés
  - [x] 2 tests d'intégration HTTP `TestClient` (direct + analysis)
- [x] Tests existants mis à jour pour refléter le nouveau comportement direct/analysis :
  - [x] `test_intent_detection.py` : tests `run_workflow` reciblés sur mode `analysis` (le mode `direct` est désormais couvert par `test_intent_workflow_branch.py`).
- [x] Non-régression globale finale : **52 tests, 0 failure, 0 error** sur 7 suites
  (`test_codestral_routing`, `test_mistral_routing`, `test_codestral_advanced`, `test_planner_fallback`, `test_logs`, `test_intent_detection`, `test_intent_workflow_branch`).
