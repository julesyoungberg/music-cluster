# Graph Report - music-cluster  (2026-11-13)

## Corpus Check
- 194 files · ~152,986 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 5, .css 2)

## Summary
- 3566 nodes · 7828 edges · 173 communities (143 shown, 30 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 674 edges (avg confidence: 0.87)
- Token cost: 257,271 input · 0 output

## Community Hubs (Navigation)
- Database Core
- Group Management
- Sample Classification
- Config Handling
- Profile Storage
- Audio Media Waveforms
- Sort Session Workflow
- Feature Extractor
- Rhythm Analysis
- Feature Layout
- API Track Routes
- Cluster Engine
- Group Naming
- Svelte UI Components
- Discovery Workflow
- Serato Crates
- Eval Metrics
- Unit Test CI Workflow
- Metadata Reading
- CLI Workflow Tests
- Lint CI Workflow
- Group Sorter
- API Tests
- CLI Group Commands
- E2E CI Workflow
- Sample Profile Tests
- UI Dev Dependencies
- Library Screen
- DJ Export Formats
- Sort Commit Planning
- Beat Tagging Rules
- Shared UI Components
- Groups Screen
- Dev and Sidecar Scripts
- Publish CI Workflow
- API Workflow Tests
- Sort Review Screen
- Pretrained Embeddings
- API Tasks and Sessions
- Embedding Space
- Tag Eval Suites
- File Organizer Undo
- Organizer Plan and Execute
- CLI Init and Import
- Waveform Player Component
- Group Detail Screen
- Library Analysis
- Sorter Tests
- Tagging Tests
- Audio File Utilities
- UI Type Definitions
- Tagging and Site Tests
- Group Filing Storage
- Eval Simulation Tests
- API Write Routes
- CLI Review and Config
- CLI Reference Docs
- Eval Runner
- DJ Export Tests
- API Request Models
- Evaluation Docs
- Eval Reports
- API Import and Fit Tests
- Discover Candidates Screen
- LLM Name Suggestion Tests
- Playwright E2E Specs
- Tauri Desktop Shell
- Eval Datasets
- API CLI Shared Context
- Tag Storage
- Embedding Fit Internals
- E2E Test Fixtures
- UI Snapshot Config
- CLI Tag Search
- Beat Detection Tests
- Audio Player Store
- Install and Requirements
- Dataset Download
- Library Analysis Tests
- CLI Sort and Apply Tests
- Discover Screen
- Analysis Result Tests
- Commit Plan Types
- Icon Generation
- UI TypeScript Config
- UI Snapshot Capture
- Tutorial and Settings Docs
- API Sort Session Tests
- UI Package Scripts
- Project Rules (CLAUDE.md)
- LLM Prompting
- UI Formatting Helpers
- Overview Screen
- New Sort Screen
- Landing Page and Testing Docs
- Product Principles Docs
- Desktop Build Steps
- Desktop Build Caching
- Site Builder
- Tag Analysis Tests
- Root Package Scripts
- UI Runtime Dependencies
- API Client
- Waveform Cache
- Settings Screen
- Sort Apply Undo Docs
- Track Tags Model
- ESLint Config
- Eval Harness Tests
- Eval Dispatch Inputs
- Synthetic Drum Rendering
- Batch Tagging
- Discovery CLI Tests
- App State Store
- Eval CI Caching
- Prettier Config
- Notifications Store
- Theme Store
- Eval Docs Overview
- Desktop Build Concurrency
- Eval CI Concurrency
- CLI Apply and Filing
- LLM Response Parsing
- Svelte Build Config
- Test Feature Fakes
- Sorter Embedding Tests
- Sort Browser Spec
- Crate Colours
- Player Test Fakes
- Self-Check Screen
- Daily Work Log
- Discovery Docs
- Publishing and CI Docs
- DJ Export Errors
- Desktop Build Matrix
- Desktop Build Triggers
- Eval Steps
- Profile Model
- Genre Tag Rules
- API Audio Tests
- Fit and Self-Check Docs
- Profile Info Routes
- Tempo Band Naming
- Duration Formatting
- Extension Parsing
- UI Test Setup
- DJ Export Docs
- Desktop Build Plan
- Desktop Release Job
- Eval CI Environment
- Eval CI Triggers
- Groove Shifting
- Vite and SvelteKit Config
- API Track Tests
- Code Layout Docs
- Eval Embedding Input
- Eval Job
- Eval Suites Input
- API Browse Tests
- SPA Layout Options
- Graphify Install Hook
- Tag Vocabulary
- Tag Winner Picking
- Tauri Cargo Manifest

## God Nodes (most connected - your core abstractions)
1. `Database` - 152 edges
2. `Config` - 108 edges
3. `run_cli()` - 54 edges
4. `EmbeddingSpace` - 50 edges
5. `Command-line reference` - 42 edges
6. `build_layout()` - 40 edges
7. `run_discovery()` - 39 edges
8. `get_context()` - 37 edges
9. `get_db()` - 35 edges
10. `FeatureExtractor` - 35 edges

## Surprising Connections (you probably didn't know these)
- `LLM naming setting` --semantically_similar_to--> `Optional LLM naming for discovery`  [INFERRED] [semantically similar]
  docs/tutorial.md → README.md
- `Files stay put until you say so (dry run, undo, four filing modes)` --semantically_similar_to--> `Dry run and undo for every filing mode`  [INFERRED] [semantically similar]
  site/index.html → CLAUDE.md
- `Two promises (you define the groups; nothing on disk changes until you say so)` --semantically_similar_to--> `User-defined groups principle`  [INFERRED] [semantically similar]
  docs/tutorial.md → README.md
- `Overlap report (per-group accuracy, confusion matrix)` --semantically_similar_to--> `Self-check (leave-one-out re-sort)`  [INFERRED] [semantically similar]
  docs/tutorial.md → README.md
- `Settings screen` --semantically_similar_to--> `Configuration (config.yaml, per-profile overrides)`  [INFERRED] [semantically similar]
  docs/tutorial.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Sorting pipeline: features, tuned space, fair distance, confident decision, learning** — readme_feature_vector, readme_lda_pca_space, readme_centroid_knn_distance, readme_confidence_margin, readme_unmatched, readme_learn_on_commit [EXTRACTED 1.00]
- **File-writing commands with preview, confirmation and undo** — docs_cli_apply, docs_cli_undo, docs_cli_groups_file, docs_cli_groups_serato, docs_tutorial_commit_dialog [INFERRED 0.85]
- **First sort loop: import, fit, check, sort, review, apply** — docs_cli_groups_import_tree, docs_cli_fit, docs_cli_check, docs_cli_sort, docs_cli_review, docs_cli_apply [EXTRACTED 1.00]
- **File safety guarantees: confirm, dry run, undo** — claude_user_confirmation_rule, claude_dry_run_and_undo, site_index_files_safety_section, claude_default_library_protection [INFERRED 0.85]
- **Sorting suites sharing the simulated-library protocol** — evaluation_sorting_fma, evaluation_sorting_fma_sub, evaluation_sorting_folders, evaluation_simulated_libraries [EXTRACTED 1.00]
- **Beat and tempo improvement loop via GiantSteps evals** — _claude_daily_work_offbeat_four_on_floor_fix, _claude_daily_work_tempo_four_thirds_error, _claude_daily_work_beat_misses, evaluation_tempo_giantsteps, eval_baselines_datasets [INFERRED 0.85]

## Communities (173 total, 30 thin omitted)

### Community 0 - "Database Core"
Cohesion: 0.04
Nodes (18): Database, _stack(), _tag_clauses(), _build_candidates(), normalize(), add_track(), test_an_undo_log_from_before_backups_gains_the_column(), test_deleting_a_collection_cascades() (+10 more)

### Community 1 - "Group Management"
Cohesion: 0.05
Nodes (26): add_tracks_to_group(), create_group(), ensure_collection(), file_url_to_path(), find_collection(), import_folder_as_group(), import_folders_as_groups(), import_playlist_as_group() (+18 more)

### Community 2 - "Sample Classification"
Cohesion: 0.04
Nodes (37): get_track_sample_profile(), parse_key(), _audio_evidence(), audio_scores(), Category, classify(), classify_many(), classify_vector() (+29 more)

### Community 3 - "Config Handling"
Cohesion: 0.06
Nodes (16): Config, expand_seeds(), is_available(), TestDefaults, TestExtractorConfig, TestPaths, TestSection, TestSetAndSave (+8 more)

### Community 4 - "Profile Storage"
Cohesion: 0.06
Nodes (32): load_sorter(), make_centres(), populated(), seeded(), test_a_sample_track_reports_its_category_and_measurements(), test_info_reports_what_was_analysed_under_each_profile(), test_the_sample_view_of_a_music_only_track_is_a_404(), test_tracks_can_be_filtered_by_profile() (+24 more)

### Community 5 - "Audio Media Waveforms"
Cohesion: 0.06
Nodes (20): cached_waveform(), clear_waveform_memory_cache(), compute_waveform(), extract_artwork(), media_type_for(), parse_range_header(), _remember(), _resize_envelope() (+12 more)

### Community 6 - "Sort Session Workflow"
Cohesion: 0.07
Nodes (23): fit_collection(), accept_all_confident(), create_session(), decide(), score_session(), test_new_one_shots_are_sorted_into_the_right_folders(), incoming(), library() (+15 more)

### Community 7 - "Feature Extractor"
Cohesion: 0.06
Nodes (16): _chroma_peak_count(), FeatureExtractor, _first_time_at_or_above(), _first_time_at_or_below(), _harmonic_support(), _onset_count(), _percussive_ratio(), aggregate_features() (+8 more)

### Community 8 - "Rhythm Analysis"
Cohesion: 0.06
Nodes (26): _align(), _analyze(), _band_onsets(), _busy_sixteenths(), _choose_downbeat(), _consistency(), _double(), _fold() (+18 more)

### Community 9 - "Feature Layout"
Cohesion: 0.07
Nodes (26): build_layout(), cosine_similarity(), describe_sample_vector(), describe_vector(), euclidean_distance(), _family_of_span(), family_weight_vector(), FeatureLayout (+18 more)

### Community 10 - "API Track Routes"
Cohesion: 0.08
Nodes (31): browse(), check_collection(), _compact_tags(), delete_collection(), delete_discovery_run(), delete_group(), delete_session(), get_candidate_members() (+23 more)

### Community 11 - "Cluster Engine"
Cohesion: 0.08
Nodes (8): ClusterEngine, select_exemplars(), blobs(), TestCluster, TestConstruction, TestEstimateK, TestGranularity, TestSelectExemplars

### Community 12 - "Group Naming"
Cohesion: 0.09
Nodes (14): audio_descriptors(), bpm_range_label(), dominant_genre(), group_stats(), _representative_bpm(), _sample_layout(), suggest_label(), TestAudioDescriptors (+6 more)

### Community 13 - "Svelte UI Components"
Cohesion: 0.06
Nodes (30): lucide-svelte, { confidence, label = '' }, tone, { group, size = 'sm' }, {
    title = 'Something went wrong',
    message = '',
    onRetry = undefined
  }, audioCount, current, directories (+22 more)

### Community 14 - "Discovery Workflow"
Cohesion: 0.07
Nodes (9): _apply_llm_names(), _default_run_name(), promote_candidate(), reject_candidate(), run_discovery(), collect_files(), TestPromoteCandidate, TestRunDiscovery (+1 more)

### Community 15 - "Serato Crates"
Cohesion: 0.08
Nodes (29): build_crate(), check_serato_dir(), _crate_name(), _crate_path(), _edited(), _fingerprint(), read_crate(), _read_manifest() (+21 more)

### Community 16 - "Eval Metrics"
Cohesion: 0.09
Nodes (23): accuracy(), auroc(), balanced_accuracy(), brier(), confusion_pairs(), expected_calibration_error(), macro_f1(), reliability() (+15 more)

### Community 17 - "Unit Test CI Workflow"
Cohesion: 0.05
Nodes (41): concurrency, cancel-in-progress, group, run, os, python-version, jobs, python (+33 more)

### Community 18 - "Metadata Reading"
Cohesion: 0.08
Nodes (11): _first_value(), _from_filename(), _parse_bpm(), _parse_year(), read_metadata(), FakeTags, TestFirstValue, TestFromFilename (+3 more)

### Community 19 - "CLI Workflow Tests"
Cohesion: 0.09
Nodes (7): run_cli(), imported(), TestCollections, TestConfigCommand, TestFoldersIAlreadyHave, TestInit, TestLibraryCommands

### Community 20 - "Lint CI Workflow"
Cohesion: 0.06
Nodes (39): concurrency, cancel-in-progress, group, run, env, NODE_VERSION, PYTHON_VERSION, jobs (+31 more)

### Community 21 - "Group Sorter"
Cohesion: 0.09
Nodes (4): GroupProfile, GroupSorter, _robust_scale(), Suggestion

### Community 22 - "API Tests"
Cohesion: 0.06
Nodes (13): client(), _tags(), test_a_group_colour_persists_without_marking_the_fit_stale(), test_audio_can_be_revalidated_rather_than_refetched(), test_dry_run_commit_writes_nothing(), test_fit_and_check(), test_saving_the_group_form_with_only_new_notes_keeps_the_fit(), test_tagging_runs_as_a_task() (+5 more)

### Community 23 - "CLI Group Commands"
Cohesion: 0.09
Nodes (19): candidates_cmd(), check(), collection_delete(), collection_list(), fit(), get_context(), groups_export(), groups_list() (+11 more)

### Community 24 - "E2E CI Workflow"
Cohesion: 0.06
Nodes (36): name, runs-on, steps, name, runs-on, steps, concurrency, cancel-in-progress (+28 more)

### Community 25 - "Sample Profile Tests"
Cohesion: 0.09
Nodes (18): embed_ast(), bass(), build_pack(), chord(), described(), extractor(), hat(), kick() (+10 more)

### Community 26 - "UI Dev Dependencies"
Cohesion: 0.08
Nodes (35): autoprefixer, concurrently, eslint, jsdom, postcss, prettier, prettier-plugin-svelte, svelte-check (+27 more)

### Community 27 - "Library Screen"
Cohesion: 0.06
Nodes (31): analyze(), analyzeOpen, analyzePath, analyzeProfile, assign(), assignOpen, beatFilter, collection (+23 more)

### Community 28 - "DJ Export Formats"
Cohesion: 0.09
Nodes (16): _absolute(), build_rekordbox_xml(), build_traktor_nml(), _is_our_export(), _is_windows_path(), _kind(), _posix_volume(), _refuse_foreign_files() (+8 more)

### Community 29 - "Sort Commit Planning"
Cohesion: 0.11
Nodes (10): utc_now(), commit_session(), _default_session_name(), _learn_from_commit(), _now(), pending_assignments(), plan_session_commit(), session_overview() (+2 more)

### Community 30 - "Beat Tagging Rules"
Cohesion: 0.12
Nodes (18): _beat_rules(), beat_scores(), danceability_score(), describe_audio(), drum_presence(), energy_score(), estimate_key(), _hits() (+10 more)

### Community 31 - "Shared UI Components"
Cohesion: 0.09
Nodes (17): svelte, vitest, SvelteKit app shell (app.html), Pre-paint theme script (mirrors lib/theme.ts), {
    icon,
    title,
    description = '',
    children
  }, {
    open = $bindable(false),
    title = '',
    description = '',
    size = 'md',
    children,
    footer
  }, widths, icons (+9 more)

### Community 32 - "Groups Screen"
Cohesion: 0.07
Nodes (32): accuracyByGroup, collection, error, EXPORT_FORMATS, exportFormat, exportGroups(), exporting, exportOpen (+24 more)

### Community 33 - "Dev and Sidecar Scripts"
Cohesion: 0.12
Nodes (11): main(), build(), main(), rust_target_triple(), smoke_test(), main(), main(), main() (+3 more)

### Community 34 - "Publish CI Workflow"
Cohesion: 0.06
Nodes (32): name, runs-on, steps, concurrency, cancel-in-progress, group, jobs, build (+24 more)

### Community 35 - "API Workflow Tests"
Cohesion: 0.07
Nodes (9): client(), collection(), fitted(), TestConfig, TestDiscovery, TestFileGroup, TestInfo, TestPrune (+1 more)

### Community 36 - "Sort Review Screen"
Cohesion: 0.07
Nodes (28): acceptConfident(), acceptedCount, commit(), commitOpen, committing, current, currentTrack, cursor (+20 more)

### Community 37 - "Pretrained Embeddings"
Cohesion: 0.12
Nodes (15): _stamp(), cache_path(), check(), _device(), embed(), decode(), load_model(), _loaded() (+7 more)

### Community 38 - "API Tasks and Sessions"
Cohesion: 0.11
Nodes (21): analyze(), worker(), AnalyzeRequest, create_discovery_run(), create_group(), create_session(), DiscoverRequest, fit_collection() (+13 more)

### Community 39 - "Embedding Space"
Cohesion: 0.15
Nodes (7): EmbeddingSpace, labelled(), TestDescribe, TestFeatureWeights, TestFit, TestSupervision, TestTransform

### Community 40 - "Tag Eval Suites"
Cohesion: 0.11
Nodes (19): key_accuracy(), key_score(), per_class(), tempo_accuracy(), beat_cases(), _beat_label_accuracy(), _beat_misses(), _by_genre() (+11 more)

### Community 41 - "File Organizer Undo"
Cohesion: 0.12
Nodes (12): _back_up(), _backup_folder(), _perform(), PlannedAction, _restore_backup(), _safe_filename(), _take_back(), undo_session() (+4 more)

### Community 42 - "Organizer Plan and Execute"
Cohesion: 0.16
Nodes (22): execute_plan(), plan_commit(), make_file(), test_a_backup_that_fails_leaves_the_file_where_it_was(), test_a_copy_that_fails_partway_leaves_the_replaced_file_in_place(), test_a_move_that_fails_after_placing_the_file_puts_both_files_back(), test_conflicts_are_skipped_by_default(), test_copy_leaves_the_original_in_place() (+14 more)

### Community 43 - "CLI Init and Import"
Cohesion: 0.12
Nodes (13): analyze(), collection_create(), discover(), eval_download(), groups_add(), groups_import_tree(), init(), progress_bar() (+5 more)

### Community 44 - "Waveform Player Component"
Cohesion: 0.08
Nodes (26): attempt, bars, canvas, container, data, draw(), duration, envelope (+18 more)

### Community 45 - "Group Detail Screen"
Cohesion: 0.08
Nodes (24): deleteOpen, description, destination, destinationOpen, error, filedSessionId, fileMode, fileOpen (+16 more)

### Community 46 - "Library Analysis"
Cohesion: 0.11
Nodes (10): analyze(), analyze_one(), cache_path(), _code_fingerprint(), default_config(), _essentia_reference(), _signature(), resolve_worker_count() (+2 more)

### Community 47 - "Sorter Tests"
Cohesion: 0.14
Nodes (14): SorterConfig, make_vector(), build_sorter(), test_a_moderately_distant_cluster_is_unmatched(), test_a_small_group_survives_beside_a_large_one(), test_distant_track_is_unmatched(), test_evaluate_flags_confusable_pairs(), test_evaluate_reports_perfect_separation() (+6 more)

### Community 48 - "Tagging Tests"
Cohesion: 0.17
Nodes (19): tag(), bar(), descriptors(), groove(), test_a_four_to_the_floor_at_95_is_not_house(), test_a_key_is_only_reported_when_there_is_one(), test_a_steady_kick_at_124_is_danceable_and_a_pad_is_not(), test_a_straight_beat_is_not_groovy_and_a_swung_one_is() (+11 more)

### Community 49 - "Audio File Utilities"
Cohesion: 0.13
Nodes (7): compute_file_checksum(), find_audio_files(), TestChecksum, TestEnsureDirectory, TestFindAudioFiles, TestGetFileInfo, touch()

### Community 50 - "UI Type Definitions"
Cohesion: 0.07
Nodes (27): AudioProfile, CandidateStats, CheckMetrics, Collection, CommitPlan, ConfusablePair, DiscoveryCandidate, DiscoveryRun (+19 more)

### Community 51 - "Tagging and Site Tests"
Cohesion: 0.13
Nodes (18): analyze(), test_duration_survives_the_round_trip(), test_a_broken_kick_keeps_its_tempo(), test_a_dotted_eighth_riff_does_not_speed_up_the_tempo(), test_a_pad_is_beatless(), test_a_slow_backbeat_is_not_doubled(), test_half_speed_tempo_readings_are_doubled(), test_rendered_grooves_are_tagged_as_drawn() (+10 more)

### Community 52 - "Group Filing Storage"
Cohesion: 0.10
Nodes (4): _with_json(), file_group(), _is_inside(), TestFileGroup

### Community 53 - "Eval Simulation Tests"
Cohesion: 0.13
Nodes (21): simulate(), SimulationSettings, _library(), test_a_failed_embedding_keeps_the_hand_feature_result(), test_an_embedding_is_scored_alone_and_with_the_features(), test_an_embedding_needs_a_sorting_suite_and_is_checked_after_the_suites(), test_beat_genre_accuracy_counts_only_unambiguous_genres(), test_beat_misses_list_wrong_beats_with_their_bar() (+13 more)

### Community 54 - "API Write Routes"
Cohesion: 0.08
Nodes (16): accept_confident(), add_group_members(), AddMembersRequest, commit_session(), CommitRequest, export_groups(), promote_candidate(), PromoteRequest (+8 more)

### Community 55 - "CLI Review and Config"
Cohesion: 0.10
Nodes (10): _coerce(), config_cmd(), eval_group(), format_confidence(), groups(), review(), review_session(), samples_group() (+2 more)

### Community 56 - "CLI Reference Docs"
Cohesion: 0.09
Nodes (25): cratewise collection create, cratewise collection delete, cratewise collection list, Command-line reference, cratewise eval download, cratewise groups add, cratewise groups list, cratewise groups remove (+17 more)

### Community 57 - "Eval Runner"
Cohesion: 0.14
Nodes (8): _probe(), _progress(), run(), _sorting(), _store(), _with_embedding(), find_tempo_column(), NotEnoughData

### Community 58 - "DJ Export Tests"
Cohesion: 0.12
Nodes (14): collect_playlists(), export_collection(), _playlist_paths(), short_key(), track_metadata(), test_an_export_refreshes_its_own_files_but_leaves_the_users_alone(), test_control_characters_in_tags_do_not_break_the_xml(), test_keys_are_written_the_way_dj_apps_read_them() (+6 more)

### Community 59 - "API Request Models"
Cohesion: 0.10
Nodes (16): BulkDecisionRequest, CollectionRequest, CollectionUpdate, ConfigUpdate, create_collection(), decide_item(), decide_items(), DecisionRequest (+8 more)

### Community 60 - "Evaluation Docs"
Cohesion: 0.15
Nodes (18): beat_misses report table (PR #33), Offbeat-grid four-to-the-floor fix (PR #34), Dataset eval baseline (datasets.md), Adding a suite (runner.py, datasets.py, test_eval.py), Eval analysis cache (~/.music-cluster-eval/data/.cache), AST pretrained embedding eval (--embedding ast), Auto-file operating points and threshold sweep, Baseline comparison with --tolerance (+10 more)

### Community 61 - "Eval Reports"
Cohesion: 0.19
Nodes (13): eval_compare(), eval_run(), build(), compare(), _fmt(), _git_commit(), _jsonable(), load() (+5 more)

### Community 62 - "API Import and Fit Tests"
Cohesion: 0.10
Nodes (3): finish(), TestErrors, TestImportAndFit

### Community 64 - "Discover Candidates Screen"
Cohesion: 0.10
Nodes (17): candidates, collection, destination, destinationOpen, error, expanded, groupName, load() (+9 more)

### Community 65 - "LLM Name Suggestion Tests"
Cohesion: 0.18
Nodes (4): suggest_names(), FakeMessages, install_fake_anthropic(), TestSuggestNames

### Community 66 - "Playwright E2E Specs"
Cohesion: 0.14
Nodes (11): @playwright/test, pile, repoRoot, uiDir, exportDir, repoRoot, crate, repoRoot (+3 more)

### Community 67 - "Tauri Desktop Shell"
Cohesion: 0.15
Nodes (7): api_base(), ApiBase, ApiProcess, free_port(), main(), shutdown(), start_api()

### Community 68 - "Eval Datasets"
Cohesion: 0.20
Nodes (10): data_dir(), fma(), folder_tree(), _giantsteps(), giantsteps_key(), giantsteps_tempo(), Item, _read_fma_genres() (+2 more)

### Community 69 - "API CLI Shared Context"
Cohesion: 0.20
Nodes (15): worker(), worker(), worker(), worker(), worker(), worker(), progress_updater(), callback() (+7 more)

### Community 70 - "Tag Storage"
Cohesion: 0.16
Nodes (7): payload(), tagged(), test_a_mood_filter_matches_whole_words_only(), test_counts_break_down_the_library(), test_deleting_a_track_deletes_its_tags(), test_saving_again_replaces(), test_untagged_and_stale_tracks_need_tagging()

### Community 72 - "E2E Test Fixtures"
Cohesion: 0.17
Nodes (8): cli_runner(), genre_folder(), library_root(), music(), workspace(), build_library(), tag(), write_track()

### Community 73 - "UI Snapshot Config"
Cohesion: 0.18
Nodes (12): run, API_PORT, apiBase, here, marker, outputDir, repoRoot, screensLog (+4 more)

### Community 74 - "CLI Tag Search"
Cohesion: 0.13
Nodes (8): _energy_bounds(), _pattern_row(), search(), _tag_line(), tags_find(), tags_show(), display_name(), TestDisplayName

### Community 75 - "Beat Detection Tests"
Cohesion: 0.14
Nodes (10): best_beat(), test_a_busy_snare_band_is_not_a_dembow_answer(), test_a_dembow_over_a_shaker_is_still_a_dembow(), test_a_late_four_on_the_floor_with_a_busy_snare_band_is_not_a_break(), test_beat_type_survives_a_wrong_downbeat(), test_each_groove_reads_as_itself(), test_four_on_the_floor_survives_a_beat_grid_off_the_kick(), test_no_drums_is_beatless_whatever_the_pattern_says() (+2 more)

### Community 76 - "Audio Player Store"
Cohesion: 0.22
Nodes (17): applyPending(), element(), followPlayback(), initial, nudge(), player, PlayerState, playTrack() (+9 more)

### Community 77 - "Install and Requirements"
Cohesion: 0.15
Nodes (12): FORMATS.md supported audio formats, Default scan extensions, librosa + FFmpeg decoding, Mixed format libraries, Prebuilt desktop installers (unsigned), Upgrading from 1.x (clustering-era migration), Frozen API Tauri sidecar, Install troubleshooting (+4 more)

### Community 78 - "Dataset Download"
Cohesion: 0.21
Nodes (7): download_fma(), download_giantsteps(), _fetch(), _md5(), _remote_size(), _unzip(), test_a_corrupt_preview_is_downloaded_again()

### Community 79 - "Library Analysis Tests"
Cohesion: 0.18
Nodes (5): analyze_files(), handle(), prune_missing(), TestAnalyzeFiles, TestPruneMissing

### Community 81 - "Discover Screen"
Cohesion: 0.12
Nodes (15): algorithm, error, load(), loading, minGroupSize, minGroupSizeTouched, newOpen, profile (+7 more)

### Community 82 - "Analysis Result Tests"
Cohesion: 0.20
Nodes (4): AnalysisResult, audio_tree(), fake_extraction(), TestAnalysisResult

### Community 83 - "Commit Plan Types"
Cohesion: 0.13
Nodes (5): CommitPlan, CommitResult, write_playlist(), get_relative_path(), TestGetRelativePath

### Community 84 - "Icon Generation"
Cohesion: 0.22
Nodes (6): icns_bytes(), ico_bytes(), main(), png_bytes(), render(), sample()

### Community 85 - "UI TypeScript Config"
Cohesion: 0.13
Nodes (14): ./.svelte-kit/tsconfig.json, compilerOptions, allowJs, checkJs, esModuleInterop, forceConsistentCasingInFileNames, lib, moduleResolution (+6 more)

### Community 86 - "UI Snapshot Capture"
Cohesion: 0.22
Nodes (13): capture(), Problem, record(), ScreenRecord, settle(), Shot, ShotOptions, slug() (+5 more)

### Community 87 - "Tutorial and Settings Docs"
Cohesion: 0.20
Nodes (14): cratewise analyze, cratewise config, Accept confident / Pre-accept, Active collection picker, App screens (Overview, Groups, Sort, Discover, Library, Settings), Group crate colour, Reference tracks and group destination, Review screen (suggested groups, queue, keyboard shortcuts) (+6 more)

### Community 89 - "UI Package Scripts"
Cohesion: 0.14
Nodes (14): scripts, build, check, check:watch, dev, format, format:check, lint (+6 more)

### Community 90 - "Project Rules (CLAUDE.md)"
Cohesion: 0.17
Nodes (7): Claude graphify skill pointer (.claude/CLAUDE.md), graphify skill (/graphify), CLAUDE.md project instructions, graphify knowledge graph rules, Ponytail: simplest solution that works, Writing rules (plain language, no em-dashes, one name per thing), MUSIC_CLUSTER_DB / MUSIC_CLUSTER_CONFIG overrides

### Community 91 - "LLM Prompting"
Cohesion: 0.22
Nodes (5): prompt_for(), _render(), _render_samples(), is_sample(), TestRender

### Community 92 - "UI Formatting Helpers"
Cohesion: 0.17
Nodes (3): confidenceClasses, formatDuration(), formatSampleLength()

### Community 93 - "Overview Screen"
Cohesion: 0.17
Nodes (12): analyzedCount, collections, current, error, fit(), fitting, load(), loading (+4 more)

### Community 94 - "New Sort Screen"
Cohesion: 0.17
Nodes (11): autoAccept, collection, error, load(), loading, newOpen, sessionName, sessions (+3 more)

### Community 95 - "Landing Page and Testing Docs"
Cohesion: 0.17
Nodes (12): Testing layers (unit, closed loop, UI unit, browser, screenshots), site.json ${key} templating (build_site.py), anthropic (optional LLM group naming), Landing page (site/index.html), Landing FAQ, Files stay put until you say so (dry run, undo, four filing modes), Newsletter and contact forms, Landing hero and how-it-works (+4 more)

### Community 96 - "Product Principles Docs"
Cohesion: 0.18
Nodes (9): CLI conventions (--collection, --profile, --workers, recursion), cratewise samples classify, Samples walkthrough, Collection profile (music vs sample), Discovery workflow for an unsorted pile, 96-dimension feature vector (125 for samples), LDA blended with PCA feature space, Sample profile analysis (one-shot measurements) (+1 more)

### Community 97 - "Desktop Build Steps"
Cohesion: 0.21
Nodes (12): steps, GH_TOKEN, steps, steps, env, id, if, name (+4 more)

### Community 98 - "Desktop Build Caching"
Cohesion: 0.17
Nodes (12): with, cache, cache-dependency-path, if-no-files-found, key, merge-multiple, name, node-version (+4 more)

### Community 99 - "Site Builder"
Cohesion: 0.26
Nodes (5): build(), load_values(), main(), placeholders(), render()

### Community 100 - "Tag Analysis Tests"
Cohesion: 0.24
Nodes (7): analyze_paths(), test_analysis_tags_music_as_it_goes(), test_samples_are_not_tagged(), test_tracks_analysed_before_tagging_can_be_tagged_later(), _write(), TestAnalyzePaths, test_tags_round_trip()

### Community 101 - "Root Package Scripts"
Cohesion: 0.18
Nodes (11): description, devDependencies, @tauri-apps/cli@^1.6.3, name, private, scripts, tauri, tauri:build (+3 more)

### Community 102 - "UI Runtime Dependencies"
Cohesion: 0.17
Nodes (12): clsx, @fontsource/ibm-plex-mono, @fontsource/ibm-plex-sans, @fontsource/ibm-plex-sans-condensed, tailwind-merge, dependencies, clsx@^2.0.0, @fontsource/ibm-plex-mono@^5.3.0 (+4 more)

### Community 103 - "API Client"
Cohesion: 0.26
Nodes (10): api, API_BASE, ApiError, del(), get(), pollTask(), post(), put() (+2 more)

### Community 104 - "Waveform Cache"
Cohesion: 0.24
Nodes (10): aborted(), cache, cachedWaveform(), fetchShared(), forgetWaveform(), inFlight, keyFor(), loadWaveform() (+2 more)

### Community 105 - "Settings Screen"
Cohesion: 0.17
Nodes (7): config, error, loading, saving, sortingFields, theme, weightFields

### Community 106 - "Sort Apply Undo Docs"
Cohesion: 0.24
Nodes (11): cratewise apply, cratewise groups file, cratewise review, cratewise sessions, cratewise sort, Sort session (open / committed), cratewise undo, Commit dialog dry run (+3 more)

### Community 107 - "Track Tags Model"
Cohesion: 0.27
Nodes (3): label_for(), TrackTags, test_energy_level_spans_one_to_ten()

### Community 108 - "ESLint Config"
Cohesion: 0.18
Nodes (10): eslint-config-prettier, @eslint/js, eslint-plugin-svelte, globals, typescript-eslint, eslint-config-prettier@^9.1.2, @eslint/js@^9.39.5, eslint-plugin-svelte@^2.46.1 (+2 more)

### Community 109 - "Eval Harness Tests"
Cohesion: 0.29
Nodes (4): cli(), main(), test_a_suite_without_its_data_is_skipped_not_fatal(), test_eval_run_writes_a_report_and_compares_against_a_baseline()

### Community 110 - "Eval Dispatch Inputs"
Cohesion: 0.20
Nodes (10): limit, trials, default, description, required, workflow_dispatch, default, description (+2 more)

### Community 111 - "Synthetic Drum Rendering"
Cohesion: 0.36
Nodes (4): _hat(), _kick(), render(), _snare()

### Community 112 - "Batch Tagging"
Cohesion: 0.20
Nodes (5): _tag_one(), tag_tracks(), handle(), job(), TaggingResult

### Community 114 - "App State Store"
Cohesion: 0.22
Nodes (7): activeCollection, activeCollectionId, collections, collectionsLoaded, groups, loadGroups(), selectCollection()

### Community 115 - "Eval CI Caching"
Cohesion: 0.22
Nodes (9): with, cache, cache-dependency-path, if-no-files-found, key, name, path, python-version (+1 more)

### Community 116 - "Prettier Config"
Cohesion: 0.22
Nodes (8): overrides, plugins, printWidth, semi, singleQuote, tabWidth, trailingComma, useTabs

### Community 117 - "Notifications Store"
Cohesion: 0.33
Nodes (8): dismiss(), Notification, NotificationKind, notifications, notify(), notifyError(), notifyInfo(), notifySuccess()

### Community 118 - "Theme Store"
Cohesion: 0.36
Nodes (7): applyTheme(), DEFAULT_THEME, isDark(), parseTheme(), storedTheme(), Theme, THEME_KEY

### Community 119 - "Eval Docs Overview"
Cohesion: 0.25
Nodes (8): cratewise eval compare, cratewise eval run, cratewise command-line tool, Desktop app (review screen with previews and waveforms), Evaluation harness (cratewise eval), Installation (installers, FFmpeg, source), music-cluster README, Performance characteristics

### Community 120 - "Desktop Build Concurrency"
Cohesion: 0.25
Nodes (7): concurrency, cancel-in-progress, group, jobs, name, permissions, contents

### Community 121 - "Eval CI Concurrency"
Cohesion: 0.25
Nodes (7): concurrency, cancel-in-progress, group, jobs, name, permissions, contents

### Community 122 - "CLI Apply and Filing"
Cohesion: 0.29
Nodes (4): apply(), echo_error(), echo_plan(), groups_file()

### Community 124 - "Svelte Build Config"
Cohesion: 0.25
Nodes (5): @sveltejs/adapter-static, @sveltejs/vite-plugin-svelte, @sveltejs/adapter-static@^3.0.0, @sveltejs/vite-plugin-svelte@^4.0.0, config

### Community 125 - "Test Feature Fakes"
Cohesion: 0.25
Nodes (5): rng(), fake_extraction(), extract_one(), incoming(), extract_one()

### Community 127 - "Sort Browser Spec"
Cohesion: 0.29
Nodes (5): audio(), newMusic, playingPast(), repoRoot, uiDir

### Community 128 - "Crate Colours"
Cohesion: 0.46
Nodes (6): CRATE_COLOR_NAMES, CRATE_COLORS, crateColor(), CrateColorName, crateColorName(), isCrateColor()

### Community 130 - "Self-Check Screen"
Cohesion: 0.25
Nodes (5): collection, error, fittedAt, loading, metrics

### Community 131 - "Daily Work Log"
Cohesion: 0.33
Nodes (7): Daily project work log, Dataset evals in GitHub Actions (cache limits), Minor-loop key misread (relative major), 4/3 tempo error fix (PR #35, SLOWER_CONSISTENCY_FLOOR), Synthetic eval baseline (synthetic.md), beats-synthetic suite, keys-synthetic suite

### Community 132 - "Discovery Docs"
Cohesion: 0.43
Nodes (7): Candidate group, cratewise candidates, cratewise discover, Environment variables (MUSIC_CLUSTER_DB, MUSIC_CLUSTER_CONFIG, ANTHROPIC_API_KEY), Discover screen (methods, candidates, keep as group), LLM naming setting, Optional LLM naming for discovery

### Community 133 - "Publishing and CI Docs"
Cohesion: 0.29
Nodes (6): Publishing guide, Homebrew cask/formula plan, PyPI trusted publishing (publish.yml), Release process (version tag, draft release), eval.yml CI workflow, CI workflows (lint, test-unit, test-e2e, build)

### Community 135 - "Desktop Build Matrix"
Cohesion: 0.29
Nodes (7): name, needs, runs-on, strategy, build, fail-fast, matrix

### Community 136 - "Desktop Build Triggers"
Cohesion: 0.29
Nodes (7): on, pull_request, push, workflow_dispatch, paths, branches, tags

### Community 137 - "Eval Steps"
Cohesion: 0.29
Nodes (7): steps, id, if, name, run, timeout-minutes, uses

### Community 139 - "Genre Tag Rules"
Cohesion: 0.29
Nodes (4): genre_from_tag(), genre_scores(), rule(), test_genre_tags_map_onto_the_vocabulary()

### Community 141 - "Fit and Self-Check Docs"
Cohesion: 0.40
Nodes (6): cratewise check, cratewise fit, cratewise groups import-tree, Overlap report (per-group accuracy, confusion matrix), Existing folders workflow (import-tree, fit, check, sort, apply), Self-check (leave-one-out re-sort)

### Community 142 - "Profile Info Routes"
Cohesion: 0.33
Nodes (3): get_info(), list_profiles(), describe_all()

### Community 147 - "DJ Export Docs"
Cohesion: 0.40
Nodes (4): cratewise groups export, cratewise groups serato, Export to DJ software (Rekordbox, Traktor, M3U), Serato crate writing

### Community 148 - "Desktop Build Plan"
Cohesion: 0.40
Nodes (5): plan, matrix, name, outputs, runs-on

### Community 149 - "Desktop Release Job"
Cohesion: 0.40
Nodes (5): release, if, name, needs, runs-on

### Community 150 - "Eval CI Environment"
Cohesion: 0.40
Nodes (5): EMBEDDING, LIMIT, REQUESTED, TRIALS, env

### Community 151 - "Eval CI Triggers"
Cohesion: 0.40
Nodes (5): on, pull_request, schedule, paths, cron

### Community 153 - "Vite and SvelteKit Config"
Cohesion: 0.40
Nodes (4): @sveltejs/kit, vite, @sveltejs/kit@^2.0.0, vite@^5.0.0

### Community 155 - "Code Layout Docs"
Cohesion: 0.50
Nodes (3): UI README, SPA layout (services/api.ts, stores, components, routes), Background task polling (/api/tasks/{id})

### Community 156 - "Eval Embedding Input"
Cohesion: 0.50
Nodes (4): default, description, required, embedding

### Community 157 - "Eval Job"
Cohesion: 0.50
Nodes (4): name, runs-on, timeout-minutes, evaluate

### Community 158 - "Eval Suites Input"
Cohesion: 0.50
Nodes (4): suites, default, description, required

### Community 160 - "SPA Layout Options"
Cohesion: 0.50
Nodes (3): prerender, ssr, trailingSlash

## Knowledge Gaps
- **44 isolated node(s):** `PATH`, `music-cluster`, `./.svelte-kit/tsconfig.json`, `Desktop app (review screen with previews and waveforms)`, `Installation (installers, FFmpeg, source)` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1410 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `lucide-svelte` connect `Svelte UI Components` to `Discover Candidates Screen`, `Groups Screen`, `Self-Check Screen`, `Sort Review Screen`, `UI Runtime Dependencies`, `Settings Screen`, `Group Detail Screen`, `Discover Screen`, `UI Dev Dependencies`, `Library Screen`, `Overview Screen`, `New Sort Screen`, `Shared UI Components`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `Database` (e.g. with `_require_collection()` and `init()`) actually correct?**
  _`Database` has 47 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PATH`, `music-cluster`, `./.svelte-kit/tsconfig.json` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Database Core` be split into smaller, more focused modules?**
  _Cohesion score 0.044568868980963045 - nodes in this community are weakly interconnected._
- **Are the 38 inferred relationships involving `Config` (e.g. with `get_config()` and `update_config()`) actually correct?**
  _`Config` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Should `Group Management` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Sample Classification` be split into smaller, more focused modules?**
  _Cohesion score 0.04442925495557074 - nodes in this community are weakly interconnected._