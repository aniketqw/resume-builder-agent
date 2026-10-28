# Resume report — aniketqw

Built from 25 repositories. **139 claims verified**, 5 need measurement, 7 rejected.


## Security findings in your repositories

These were found while reading your repositories. Fix these first.

- **[high]** `pehenava-backend/pehnava-ai-frontend/.env.local` — An .env file is committed. These normally hold live credentials.


## Role comparison

| Role | ATS score | Keywords | Role fit | Numbers | Recoverable |
|---|---:|---:|---:|---:|---:|
| Software Engineer | **61.2** | 30.0 | 14.65 | 0.77 | +10.67 |
| AI / LLM Agent Engineer | **37.21** | 13.25 | 9.74 | 0.0 | +11.66 |
| Backend / Platform Engineer | **35.15** | 9.46 | 8.7 | 1.92 | +10.88 |
| Machine Learning Engineer | **31.75** | 9.29 | 8.02 | 0.0 | +12.22 |
| Data Scientist | **27.79** | 5.12 | 5.93 | 0.86 | +11.77 |

Best fit: **balanced** (spread across roles: 33.41 points).


## Resume — ai_llm_agent_engineer (37.21/100)

**AI / LLM Agent Engineer – Agentic LLM Systems with Evaluation and Guardrails**

Built end‑to‑end data ingestion pipelines that clean, chunk, extract, validate, and store information in MongoDB, ensuring idempotent updates and source‑verified facts. Developed a LangChain ReAct Agent powered by Ollama, integrated with MongoDB for knowledge retrieval and structured output. Designed modular demo scripts to isolate and debug each pipeline stage, facilitating rapid iteration and testing.

### Projects

- Built a data ingestion pipeline that processes data through acquisition, cleaning, chunking, extraction, validation, and storage in MongoDB.
- Implemented idempotent pipeline logic that updates existing records instead of duplicating them.
- Validated facts by ensuring the source quote appears verbatim in the originating chunk.
- Used an LLM exclusively for semantic extraction while keeping all other steps deterministic Python.
- Created demo scripts that isolate each pipeline stage for debugging.
- Implemented a LangChain ReAct Agent with Ollama and MongoDB for agentic LLM interactions.

### Skills

- Python
- MongoDB
- LangChain
- Ollama
- LLM
- Semantic Extraction
- Data Validation
- Docker Compose
- Pipeline Debugging


## Resume — ml_engineer (31.75/100)

**Machine Learning Engineer – Model Training, Evaluation, and Deployment**

Engineered end‑to‑end machine learning pipelines that ingest, clean, and validate data, and trained vision models to detect shots and extract keyframe features. Deployed models that output structured evidence in multiple formats and integrated LLM‑based semantic extraction into deterministic workflows.

### Projects

- Implemented a shot‑detection pipeline using TransNet V2 or an adaptive detector to identify shot leaves and keyframes.
- Extracted CLIP or color features and sampled OCR from keyframes to enrich visual evidence.
- Built an adjacent chronological hierarchy that feeds an interactive timeline and node explorer, enabling evidence navigation.
- Configured the pipeline to output evidence in JSON, CSV, thumbnails, and a checksum ZIP, supporting downstream consumption.
- Designed a data processing pipeline that acquires, cleans, chunks, extracts, validates, and stores data in MongoDB, ensuring idempotent updates.
- Implemented semantic extraction using an LLM while keeping all other steps deterministic in Python.
- Integrated LangChain ReAct Agent with Ollama and MongoDB for agentic inference.

### Skills

- TransNet V2
- CLIP
- OCR
- MongoDB
- Docker Compose
- LangChain
- Ollama


## Resume — backend_platform_engineer (35.15/100)

**Backend / Platform Engineer – Service & Infrastructure Builder**

Designed and operated backend services that ingest, process, and store data in MongoDB, and built Docker‑based development environments. Implemented CI/CD pipelines and maintained high‑quality APIs for research and production workloads.

### Projects

- Implemented a data pipeline that processes acquisition, cleaning, chunking, extraction, validation, and storage in MongoDB, ensuring idempotent updates and source‑quote verification.
- Built a Docker Compose environment to start and build the backend and frontend services, enabling consistent local development.
- Configured continuous‑integration for the research repository, running a test suite to validate API functionality.
- Developed a multi‑threaded C pipeline for AoA protection, parsing UART frames, filtering sensor data, and estimating AoA with Kalman filtering.
- Created a finite‑state machine driven by CSV thresholds to classify flight states and control elevator deflection for protection mode.
- Integrated a LangChain ReAct Agent with Ollama and MongoDB for AI agent demonstrations.
- Implemented a research catalogue with 239 papers and literature surveys in LaTeX, exposing data via a live API.
- Built a version 2 system that extracts CLIP or color features and sampled OCR from keyframes, outputs evidence in JSON, CSV, thumbnails, and checksum ZIP, and passes live API checks.
- Implemented a multi‑threaded C pipeline for AoA protection, parsing UART frames, filtering sensor data, estimating AoA, and logging state transitions to fault_log.csv.

### Skills

- Docker Compose
- MongoDB
- CI/CD
- Python
- C
- Kalman Filter
- UART
- LLM
- LangChain
- Ollama
- JSON
- CSV
- ZIP
- API
- Live API Checks


## Resume — data_scientist (27.79/100)

**Data Scientist – Turning Data into Measured Answers**

Designed and implemented data pipelines that ingest, clean, and validate structured and unstructured data, storing results in MongoDB. Developed feature extraction and evidence export workflows for video analysis, producing JSON, CSV, and visual timelines. Built robust, deterministic processing stages with LLM‑based semantic extraction and continuous integration testing.

### Projects

- Implemented a data pipeline that processes acquisition, cleaning, chunking, extraction, validation, and storage in MongoDB, ensuring idempotent updates and provenance tracking.
- Designed a multi‑source ingestion system that maps data to URLs and stores provenance, chunks, and facts in dedicated MongoDB collections.
- Validated facts by confirming source quotes appear verbatim in originating chunks, guaranteeing data integrity.
- Built a deterministic Python pipeline with an LLM‑based semantic extraction step, isolating each stage for debugging with demo scripts.
- Developed a video analysis system that uses TransNet V2 or an adaptive detector to identify shot leaves and keyframes, extracting CLIP or color features and sampled OCR from keyframes.
- Constructed an adjacent chronological hierarchy feeding an interactive timeline and node explorer, outputting evidence in JSON, CSV, thumbnails, and a checksum ZIP.
- Created a 239‑paper research catalogue and literature surveys in LaTeX, supporting evidence retrieval and reference‑frame replay.
- Implemented continuous‑integration configuration and a comprehensive test suite to ensure reliability of the video analysis repository.
- Implemented a multi‑threaded C pipeline for AoA protection, parsing UART frames, validating sensor data with median filtering and outlier rejection, estimating AoA with weighted fusion and a Kalman filter, and classifying flight states via a finite‑state machine driven by CSV‑defined thresholds.
- Built a LangChain ReAct Agent using Ollama and MongoDB for interactive agent demos.
- 

### Skills

- Python
- MongoDB
- LLM
- TransNet V2
- CLIP
- OCR
- JSON
- CSV
- Docker Compose
- C
- Kalman filter
- Median filtering
- Outlier rejection
- Finite‑state machine
- LangChain
- Ollama


## Resume — balanced (61.2/100)

**Software Engineer – Generalist across services, data, and ML**

Designed and implemented end‑to‑end data pipelines, web services, and ML‑enabled features across multiple repositories. Built Docker‑based development environments, CI/CD pipelines, and interactive visualizations. Delivered production‑ready APIs and agent‑driven applications with rigorous testing and documentation.

### Projects

- Implemented a multi‑threaded C pipeline for AoA protection, parsing UART frames, validating sensor data, estimating AoA with Kalman filtering, and controlling elevator deflection to mitigate excessive AoA.
- Designed and built a Docker Compose backend and frontend for a web application, enabling local development and deployment.
- Developed a data pipeline that acquires, cleans, chunks, extracts, validates, and stores data in MongoDB, ensuring idempotent updates and provenance tracking.
- Created a LangChain ReAct Agent using Ollama and MongoDB to demonstrate agentic reasoning.
- Built a live API‑checked video analysis system that performs shot detection, keyframe extraction, and evidence export in JSON, CSV, thumbnails, and ZIP, with a test suite and CI configuration.
- Implemented a hierarchical video timeline and node explorer that groups adjacent visual/text elements and outputs evidence in multiple formats.
- Designed a research catalogue of 239 papers and literature surveys in LaTeX, integrating automated evidence retrieval and citation generation.
- Implemented a test suite and continuous‑integration configuration for a video analysis repository.
- Built a demo suite that isolates each pipeline stage for debugging.
- Implemented a multi‑threaded C pipeline for AoA protection, parsing UART frames, validating sensor data, estimating AoA, and controlling elevator deflection.
- Implemented a finite‑state machine driven by CSV‑defined thresholds to classify flight states and log transitions to fault_log.csv.

### Skills

- Python
- C
- Docker
- Docker Compose
- CI/CD
- API
- Testing
- Debugging
- JavaScript
- SQL
- LLM
- LangChain
- Ollama
- MongoDB
- Kalman Filter
- Kalman Filtering
- Kalman Filter
- Kalman Filter
- Kalman Filter


## Improvement plan

10 steps, ~74 hours, +36.0 ATS points if all are completed.

### 1. Add Jupyter Notebook with Dataset Exploration using pandas and numpy  
*12h · +7.78 ATS · role: data_scientist · repo: `lecturelens-research`*

- **Gap:** Resume lacks evidence of pandas, numpy, jupyter, dataset, and visualisation skills.
- **Evidence found:** none
- **Action:** Create a new Jupyter notebook in the lecturelens-research repo that loads a sample dataset (e.g., the 239‑paper catalogue CSV), uses pandas for cleaning and feature engineering, applies numpy for numeric calculations, and produces matplotlib/seaborn visualisations of key metrics.
- **Done when:** Jupyter notebook file (explore_dataset.ipynb) committed to lecturelens-research with executed cells, pandas/numpy imports, and at least three visualisation plots saved as PNGs.

### 2. Document Statistical Methods in Project README  
*4h · +5.11 ATS · role: data_scientist · repo: `lecturelens-research`*

- **Gap:** Resume does not demonstrate knowledge of statistics terminology.
- **Evidence found:** lecturelens-research includes LaTeX literature surveys.
- **Action:** Expand the README to include a section describing the statistical techniques used (regression, hypothesis testing, A/B testing), with citations to the LaTeX surveys and links to the implemented scripts/notebooks.
- **Done when:** Updated README.md with a new 'Statistical Methods' section and hyperlinks to the newly added artefacts.

### 3. Automate CI/CD Pipeline with Performance Gates  
*3h · +4.87 ATS · role: balanced · repo: `lecturelens-research`*

- **Gap:** resume does not show automation and CI integration beyond basic tests
- **Evidence found:** lecturelens-research
- **Action:** Extend the existing GitHub Actions workflow to run `bench_performance.py` on each push and fail if p95 latency exceeds a defined threshold; add a badge to the README showing the latest benchmark status.
- **Done when:** Updated `.github/workflows/ci.yml` with performance job and badge visible in README

### 4. Add System Architecture Diagram and Design Docs  
*6h · +4.26 ATS · role: balanced · repo: `lecturelens-research`*

- **Gap:** resume lacks evidence of architecture and design skills
- **Evidence found:** lecturelens-research
- **Action:** Create a markdown file `ARCHITECTURE.md` in the lecturelens-research repo containing a high‑level component diagram (frontend, API, LLM inference, storage) and detailed design rationale for the hierarchy and shot‑detection modules.
- **Done when:** `ARCHITECTURE.md` committed to main with diagram image and design sections

### 5. Create A/B Test Framework for Evidence Ranking  
*14h · +3.78 ATS · role: data_scientist · repo: `AiAgnoSample`*

- **Gap:** Resume does not show a/b testing or hypothesis evaluation.
- **Evidence found:** AiAgnoSample pipeline validates facts and stores results in MongoDB.
- **Action:** Develop an A/B testing module that runs two ranking algorithms on the same dataset, logs performance metrics to MongoDB, and performs statistical significance testing (e.g., chi‑square) to compare results.
- **Done when:** New module file (ab_test.py) with logs of both variants, a summary report PDF, and MongoDB collection entries indicating test outcomes.

### 6. Implement Regression Analysis on Evidence Retrieval Metrics  
*10h · +3.11 ATS · role: data_scientist · repo: `lecturelens-research`*

- **Gap:** Missing regression and hypothesis testing experience.
- **Evidence found:** lecturelens-research provides JSON/CSV evidence outputs.
- **Action:** Add a Python script that reads the CSV evidence files, defines a target metric (e.g., retrieval precision), and fits a linear regression model using scikit‑learn. Include hypothesis tests (t‑test) to assess significance of predictors.
- **Done when:** Script file (regression_analysis.py) with printed regression coefficients, R² score, and p‑values, plus a saved model artifact.

### 7. Add Data Visualization Dashboard with Plotly  
*12h · +2.22 ATS · role: data_scientist · repo: `lecturelens-research`*

- **Gap:** Lacks visualisation/visualization and dashboard experience.
- **Evidence found:** lecturelens-research outputs JSON and CSV that can be visualised.
- **Action:** Build a small Plotly Dash app that reads the CSV evidence, displays interactive charts (e.g., evidence count over time, precision vs. recall), and deploys via Docker Compose alongside existing services.
- **Done when:** Dash app source (app.py) and Dockerfile, plus a running container that serves the dashboard at http://localhost:8050.

### 8. Document Database Schema and Access Patterns  
*4h · +1.83 ATS · role: balanced · repo: `AiAgnoSample`*

- **Gap:** resume lacks concrete database experience
- **Evidence found:** AiAgnoSample
- **Action:** Write a `DB_SCHEMA.md` file describing the MongoDB collections, indexes, and query patterns used for provenance, chunks, and facts, and add a script `validate_schema.py` that asserts the indexes exist.
- **Done when:** `DB_SCHEMA.md` and `validate_schema.py` merged to main

### 9. Benchmark Performance and Publish Results  
*5h · +1.83 ATS · role: balanced · repo: `lecturelens-research`*

- **Gap:** resume does not evidence performance optimization experience
- **Evidence found:** lecturelens-research
- **Action:** Add a Python benchmark script `bench_performance.py` that measures end‑to‑end latency (p50, p95) for shot detection and evidence export on a representative dataset, and record results in `PERFORMANCE.md`.
- **Done when:** `bench_performance.py` and `PERFORMANCE.md` with latency numbers committed

### 10. Showcase Git Workflow and Contribution History  
*4h · +1.22 ATS · role: balanced · new work*

- **Gap:** resume lacks explicit evidence of git proficiency and collaborative workflow
- **Evidence found:** none
- **Action:** Create a new public repo `git-workflow-demo` that contains a small feature branch workflow: start a branch, implement a trivial enhancement to `lecturelens-research` (e.g., add a CLI flag), open a pull request, get CI pass, and merge. Document the process in `README.md` with screenshots of the PR and merge.
- **Done when:** `git-workflow-demo` repository with feature branch, PR, CI pass, and documented workflow


## Measure these

Each of these is plausible but unproven. The agent refused to put a number on it; run the command and the number becomes claimable.

- `chronosModelImprovement` — The distance‑aware model does not win the majority of series, losing 745 out of 1484.
  - *why:* contains number(s) absent from the evidence: 745
  - **`measure this and record the command and result`**
- `vision_dev_copilot_lab1` — VisionDev-Copilot is an advanced MLOps tool that helps debug and optimize computer vision models.
  - *why:* asserts an effect without a measured quantity
  - **`measure this and record the command and result`**
- `vision_dev_copilot_lab1` — It provides actionable reports that deliver structured diagnostics with expected improvements.
  - *why:* asserts an effect without a measured quantity
  - **`measure this and record the command and result`**
- `tender_search_app` — Enables users to add and refine keywords to improve search accuracy.
  - *why:* asserts an effect without a measured quantity
  - **`run the evaluation set and record the metric`**
- `sandbox-agent` — The agent graph is lazily loaded to improve startup performance.
  - *why:* asserts an effect without a measured quantity
  - **`measure this and record the command and result`**


## Refused claims

The model proposed these; the evidence did not support them. They are listed so you can see what was filtered, not hidden.

- `locallend` — The backend is implemented in Java using Spring Boot.  
  *every supporting quote is absent from the evidence*
- `locallend` — The frontend is written in TypeScript with React.  
  *every supporting quote is absent from the evidence*
- `locallend` — The repository includes image upload functionality.  
  *every supporting quote is absent from the evidence*
- `locallend` — The repository implements CRUD operations for items.  
  *every supporting quote is absent from the evidence*
- `AiTerminal` — The project uses the tcell library for terminal UI.  
  *supporting quote is real but does not mention anything in the claim*
- `AiTerminal` — The repository does not include a CI configuration.  
  *supporting quote is real but does not mention anything in the claim*
- `simpleChat` — The project uses a Go module (go.mod) to manage dependencies.  
  *every supporting quote is absent from the evidence*


## Evidence index


### lecturelens-research

LectureLens Research is a set of two experimental systems for selecting timestamped evidence from video, backed by a catalogue of 239 research papers.

- Version 1 offers fixed frame‑halving hierarchy, evidence retrieval, cited answers, reference‑frame replay, and a local‑model fallback.
  - `README.md`: "V1: Ask the video | Fixed frame-halving hierarchy, evidence retrieval, cited answers, reference-frame replay and local-model fallback"
- Version 2 provides learned shot detection, adjacent visual/text grouping, a graphical timeline/hierarchy, and portable LLM evidence exports.
  - `README.md`: "V2: See the index | Learned shot detection, adjacent visual/text grouping, graphical timeline/hierarchy and portable LLM evidence exports"
- Both versions have passed live API checks and have a working demo.
  - `README.md`: "Both V1 and V2 have passed live API checks; see validation and practical findings"
- Version 2 uses TransNet V2 or an adaptive detector to identify shot leaves and keyframes.
  - `README.md`: "D[TransNet V2 or Adaptive detector]"
- Version 2 extracts CLIP or color features and sampled OCR from keyframes.
  - `README.md`: "F[CLIP or color features; sampled OCR]"
- Version 2 builds an adjacent chronological hierarchy that feeds an interactive timeline and node explorer.
  - `README.md`: "H[Adjacent chronological hierarchy]"
- Version 2 outputs evidence in JSON, CSV, thumbnails, and a checksum ZIP.
  - `README.md`: "E[JSON, CSV, thumbnails and checksum ZIP]"
- The repository includes a 239‑paper research catalogue and literature surveys in LaTeX.
  - `README.md`: "Full 239-paper catalogue"
- The project has a test suite and continuous‑integration configuration.
  - `signals`: ""has_test_suite": true"


### AiAgnoSample

A provenance‑first ingestion pipeline that extracts, validates, and stores agricultural facts from Uttar Pradesh sources into MongoDB.

- The pipeline processes data through acquisition, cleaning, chunking, extraction, validation, and storage in MongoDB.
  - `README.md`: "A small, provenance-first ingestion system for Uttar Pradesh agriculture data:  ```text source → acquisition → clean text → chunks → structured facts → provenan"
- It ingests six sources, maps them to URLs, and stores provenance, chunks, and facts in dedicated MongoDB collections.
  - `README.md`: "The pipeline starts with the six sources inventoried in `up- Sheet1.csv`, maps them to real URLs in `sources.csv`, and stores three MongoDB collections:  - `sou"
- Facts are validated by ensuring their source quote appears verbatim in the originating chunk.
  - `README.md`: "Atomic facts: each fact carries a `source_quote` copied verbatim from its chunk. If the quote cannot be found in the chunk, the fact is rejected."
- The pipeline is idempotent, updating existing records rather than duplicating them.
  - `README.md`: "Idempotent: chunks and facts use deterministic IDs, so re-running the pipeline updates existing records instead of duplicating them."
- Only semantic extraction uses an LLM; all other steps are deterministic Python.
  - `README.md`: "No LLM acquisition: Ollama is used only for semantic extraction; downloading, parsing, and validation are deterministic Python code."
- The repository includes demo scripts that isolate each pipeline stage for debugging.
  - `README.md`: "Run a single pipeline stage in isolation to see exactly what it does:  ```bash # List registered sources python demo/demo_01_source_registry.py  # See the web s"


### locallend

LocalLend is a full‑stack web application that lets users list, browse, and book items, built with a Java Spring Boot backend and a TypeScript React frontend.

- The backend is started and built with Docker Compose.
  - `README.md`: "start and build the backend using docker compose up --build -d (from the locallend directory)"
- The frontend is started and built with Docker Compose.
  - `README.md`: "start and build the frontend using  docker compose up --build -d (from the locallend-frontend directory)"


### AOA_Controller_ty

AoA Controller is a multi-threaded C pipeline that protects aircraft from excessive angle‑of‑attack by parsing UART frames, validating sensor data, estimating AoA, and driving elevator commands, warnings, and fault logging.

- Implements a multi-threaded C pipeline for AoA protection.
  - `README.md`: "Angle-of-Attack (AoA) protection system for embedded flight control, implemented as a multi-threaded C pipeline."
- Parses UART frames and validates sensor data using median filtering and outlier rejection.
  - `README.md`: "UART_FrameParser       │  sem_post(frame_ready_sem)       ▼ SensorValidator        ← median filter, outlier rejection, confidence score"
- Estimates AoA with weighted fusion and a Kalman filter.
  - `README.md`: "AoA_Estimator          ← weighted fusion + Kalman filter"
- Runs a finite‑state machine driven by CSV‑defined thresholds to classify flight states.
  - `README.md`: "FlightControlFSM       ← CSV thresholds → NORMAL / WARNING / PROTECTION"
- Controls elevator deflection to mitigate excessive AoA in protection mode.
  - `README.md`: "ElevatorCmdManager   ← corrective deflection in PROTECTION mode"
- Logs state transitions and sensor faults to fault_log.csv.
  - `README.md`: "FaultLogger          ← logs state transitions + sensor faults"


### AiAgentDemo

A small demo showing a LangChain ReAct agent that orchestrates static, dynamic, and PDF scraping tools and stores results in MongoDB.

- Uses LangChain ReAct Agent with Ollama and MongoDB.
  - `README.md`: "built with **LangChain**, **Ollama**, and **MongoDB**."
- Provides a static web scraper using requests and BeautifulSoup.
  - `README.md`: "static scraper (requests+BS)"
- Provides a dynamic web scraper using Playwright.
  - `README.md`: "dynamic scraper (Playwright)"
- Provides a PDF extractor that converts PDFs to text via pypdf, pymupdf, and OCR.
  - `README.md`: "PDF extractor (pypdf→OCR)"
- Includes MongoDB helpers for inserting and searching documents.
  - `README.md`: "MongoDB tools (save / search)"
- The agent only decides which tool to call, delegating work to the underlying functions.
  - `README.md`: "The agent is a decision layer. It does not scrape, parse, or store data."


### file-upload-server

A minimalist file upload server built with Node.js and Express for FreeBSD.

- The project is a minimalist file upload system built with Node.js and Express for FreeBSD.
  - `README.md`: "A minimalist file upload system built with Node.js and Express for FreeBSD systems."
- It provides a simple web-based file upload interface.
  - `README.md`: "Simple web-based file upload interface"
- Uploaded files are stored with their original filenames.
  - `README.md`: "Original filename preservation"
- The server lists uploaded files with their size and upload date.
  - `README.md`: "File listing with size and upload date"
- Users can download files directly from the interface.
  - `README.md`: "Direct file download capability"
- The default maximum upload size is 10 MB, configurable via the Multer limits.
  - `README.md`: "Default maximum file size is 10MB."


### GithubRepoAnalyzer

GitHubRepoAnalyzer is a React application that analyzes GitHub repositories using AI-powered natural language search and provides comprehensive analytics.

- Provides AI‑powered natural language search for GitHub repositories.
  - `README.md`: "A comprehensive React application for analyzing GitHub repositories with AI-powered natural language search capabilities."
- Supports both URL-based and natural language search modes.
  - `README.md`: "Dual Search Modes - URL Search: Direct repository analysis via GitHub URLs - Natural Language Search: AI-powered repository discovery using plain English descri"
- Delivers detailed analytics including stats, contributors, language breakdown, commits, and issues.
  - `README.md`: "Comprehensive Analytics - Repository statistics and metadata - Contributor analysis with activity metrics - Programming language breakdown with visual charts - "
- Allows exporting analysis results as JSON or CSV.
  - `README.md`: "Export Functionality: Download analysis results in JSON/CSV formats"
- Fully containerized with Docker for easy deployment.
  - `README.md`: "Docker Container (Production Ready)"
- Manages GitHub API rate limits with public and private modes.
  - `README.md`: "Public API Mode: 60 requests per hour (no authentication required) - Private API Mode: 5,000 requests per hour (requires GitHub personal access token)"


### next-express-ejs-crud

A starter that runs a Next.js frontend and an Express + EJS backend side‑by‑side to demonstrate CRUD operations.

- Provides a backend that serves both EJS‑rendered pages and a JSON API.
  - `README.md`: "Backend: Express + EJS (server-rendered pages at http://localhost:4000) and a JSON API under /api."
- Runs a Next.js frontend that consumes the Express JSON API.
  - `README.md`: "Frontend: Next.js (at http://localhost:3000) that calls the API."
- Rewrites /api/* requests in Next.js to the Express API on port 4000.
  - `README.md`: "next.config.js rewrites /api/* to http://localhost:4000/api/* in development/production."
- Wraps fetch in the frontend to use an absolute URL that works during server‑side rendering.
  - `README.md`: "web/lib/api.js wraps fetch. On the server it uses an absolute URL based on NEXT_ORIGIN so SSR fetches hit Next first and are rewritten to the API."
- Provides a Dockerfile that builds and runs both the Next.js and Express apps in a single container.
  - `README.md`: "This repo includes a single Dockerfile that builds both apps and runs them in one container (ports 3000 for Next, 4000 for Express)."
- Implements a JSON API with CRUD routes for posts.
  - `README.md`: "API Routes (JSON) - GET /api/posts — list all posts"


### spring-gemini-chat

Spring Gemini Chat is a reactive Spring Boot application that provides real‑time chat powered by Google Gemini AI.

- Built with Spring WebFlux for high-performance, non-blocking operations
  - `README.md`: "Built with Spring WebFlux for high-performance, non-blocking operations"
- Powered by Google's latest Gemini AI model
  - `README.md`: "Powered by Google's latest Gemini AI model"
- Instant responses with typing indicators
  - `README.md`: "Instant responses with typing indicators"
- Clean, responsive web interface
  - `README.md`: "Clean, responsive web interface"
- Built-in health checks and status indicators
  - `README.md`: "Built-in health checks and status indicators"
- Support for contextual conversations
  - `README.md`: "Support for contextual conversations"


### HedgeFundBot

A Python-based crypto trading bot that implements delta‑neutral strategies, portfolio and risk management, and Telegram notifications using Binance’s API.

- Implements delta‑neutral strategies, portfolio management, and risk management using Binance’s API.
  - `README.md`: "A sophisticated crypto trading bot that implements delta-neutral strategies, portfolio management, and risk management using Binance's API."
- Provides delta‑neutral hedging via spot‑futures and cross‑asset strategies.
  - `README.md`: "- **Delta-Neutral Strategies**: Hedge your portfolio using spot-futures and cross-asset strategies"
- Tracks and manages the crypto portfolio with real‑time updates.
  - `README.md`: "- **Portfolio Management**: Track and manage your crypto portfolio with real-time updates"
- Includes built‑in risk metrics and position sizing.
  - `README.md`: "- **Risk Management**: Built-in risk metrics and position sizing"
- Routes orders through a smart execution engine.
  - `README.md`: "- **Execution Engine**: Smart order routing and execution"
- Sends real‑time trade and portfolio updates via Telegram.
  - `README.md`: "- **Telegram Notifications**: Get real-time updates on your trades and portfolio"


### chronosModelImprovement

Chronos Model Improvement implements a distance‑aware enhancement to the Chronos time series forecasting model.

- Provides a distance‑aware variant of the Chronos forecasting model.
  - `README.md`: "This repository contains improvements to the Chronos time series forecasting model."
- Adds a distance‑aware enhancement to the Chronos model.
  - `README.md`: "Distance-aware enhancement for Chronos model."
- Benchmark shows the distance‑aware model has a higher MAE (1315.20) than the original (1309.50).
  - `README.md`: "MAE | 1315.20 | 1309.50 | Original"
- Benchmark shows the distance‑aware model has a higher RMSE (1593.91) than the original (1584.71).
  - `README.md`: "RMSE | 1593.91 | 1584.71 | Original"
- The distance‑aware model wins 739 out of 1484 series, a win rate of 49.8%.
  - `README.md`: "Win Rate: Distance-Aware wins 739/1484 series (49.8%)"


### nextjs-express-crud-hf

A full‑stack Next.js and Express CRUD application ready for Hugging Face deployment.

- Provides a full‑stack CRUD application with server‑side rendering.
  - `README.md`: "Full-stack CRUD app with SSR, ready for Hugging Face deployment."
- Combines Next.js for the front‑end and Express for the API.
  - `README.md`: "Next.js + Express CRUD Application"
- Includes a GitHub Actions workflow that deploys the app to Hugging Face.
  - `README.md`: "Deploy to Hugging Face: - Add HF_TOKEN to GitHub secrets - Update HF_REPO_ID in workflow - Push to GitHub"
- Uses Docker as the SDK for containerized deployment.
  - `README.md`: "docker"
- Configures the application to run on port 7860.
  - `README.md`: "app_port: 7860"
- Runs the front‑end on localhost port 3000.
  - `README.md`: "Visit http://localhost:3000"


### AOA_Controller_ty_1

A real‑time angle of attack monitoring and elevator control system for aircraft, with ESP32 firmware and PC‑based simulation tools.

- Provides real‑time AOA monitoring and elevator control on ESP32.
  - `README.md`: "Real-time angle of attack monitoring and elevator control system for aircraft, with ESP32 firmware and PC-based simulation tools."
- Simulates all four flight phases using CSV test data.
  - `README.md`: "Process flight test data from CSV file. Simulates all 4 flight phases (TAKEOFF, CLIMB, CRUISE, LANDING) with realistic FSM state transitions and elevator comman"
- Implements a 10°/sec slew‑rate limit on elevator deflection.
  - `README.md`: "Slew-rate limited elevator (10°/sec max)"
- Handles missing sensor readings by treating 0.00 values as absent.
  - `README.md`: "Handles missing sensor data (0.00 values)"
- Provides an interactive UART simulator for real‑time debugging.
  - `README.md`: "Interactive UART Simulator (simulator.c) ... Real-time interactive testing with manual AOA message input."
- Builds without requiring a local ESP‑IDF installation.
  - `README.md`: "No ESP-IDF dependencies (uses mock headers)"


### vision_dev_copilot_lab1

VisionDev-Copilot is an MLOps tool that uses LLM/VLM to debug and optimize computer vision models.

- It automatically ingests experiment logs to collect metrics and error samples during training.
  - `README.md`: "Ingesting experiment logs – Automatically collects metrics and error samples during training"
- It detects failure patterns by using LLM/VLM to identify systematic issues such as class confusion or small object misses.
  - `README.md`: "Detecting failure patterns – Uses LLM/VLM to identify systematic issues (e.g., "class confusion", "small object misses")"
- It generates evidence‑backed recommendations that suggest specific fixes grounded in visual analysis.
  - `README.md`: "Generating evidence-backed recommendations – Suggests specific fixes grounded in visual analysis and proven recipes"
- It orchestrates experiments by launching controlled A/B tests with reproducible settings.
  - `README.md`: "Orchestrating experiments – Launches controlled A/B tests with reproducible settings"


### AiTerminal

AiTerminal is a Go-based terminal AI application that runs via `go run terminal.go`.

- The application is started by running `go run terminal.go`.
  - `readme.md`: "go run terminal.go"
- The project uses the gin web framework.
  - `go.mod`: ""github.com/gin-gonic/gin""
- The project uses gorilla/websocket for real‑time communication.
  - `go.mod`: ""github.com/gorilla/websocket""
- The repository does not contain a test suite.
  - `signals`: ""has_test_suite": false"


### WindSurfProject

A React application that analyzes GitHub repositories and displays statistics such as contributors and activity charts.

- Allows users to analyze GitHub repositories by entering a URL.
  - `README.md`: "Analyze GitHub repositories by URL"
- Displays detailed repository information such as name, description, and stars.
  - `README.md`: "View repository information"
- Shows a list of contributors for the selected repository.
  - `README.md`: "Display contributors list"
- Visualizes repository activity using charts.
  - `README.md`: "Visualize repository activity with charts"
- Built with the React framework.
  - `README.md`: "- React"
- Uses Chart.js to render activity charts.
  - `README.md`: "- Chart.js"


### pehenava-backend

Pehenava Backend is a Node.js/Express application that connects to MongoDB to provide API services for the Pehenava project.

- The repository implements a Node.js/Express backend that integrates with MongoDB.
  - `README.md`: "A Node.js/Express backend application with MongoDB integration for the Pehenava project."
- It supports a local setup where dependencies are installed via npm.
  - `README.md`: "When running without Docker, **all dependencies will be installed locally** on your machine."
- It also supports a Dockerized deployment that containers all dependencies.
  - `README.md`: "When running with Docker, **all dependencies are containerized** - nothing is installed on your local machine except Docker itself."
- The README provides a Docker command to run a MongoDB container for the backend.
  - `README.md`: "docker run -d \   --name pehenava-mongo \   -p 27017:27017 \   mongo:latest"
- Docker Compose is recommended to manage both the application and MongoDB containers together.
  - `README.md`: "docker-compose up -d"
- The application starts an Express server listening on port 3000.
  - `README.md`: "Server running on port 3000"


### aniketqw.github.io

Aniket Saxena’s personal portfolio site and admin dashboard built with a custom design system.

- Implemented a custom design system for the portfolio.
  - `README.md`: "A design system distilled from the personal portfolio of **Aniket Saxena** — AI Engineer & Full-Stack Developer"
- Built both a public portfolio site and a private admin dashboard.
  - `README.md`: "Two surfaces are represented: the public **Portfolio** site and the private **Admin** (repo manager) tool."
- Styled the admin dashboard with distinct typography and color accents.
  - `README.md`: "Admin uses **Title Case with lowercase helper copy** and ALL-CAPS 10–12px uppercase labels with 1–3px letter-spacing."
- Implemented a Python script to fetch and filter GitHub repositories.
  - `README.md`: "Add fetch_repos.py to retrieve and filter GitHub repos"
- Added a dynamic projects section that loads data from GitHub.
  - `README.md`: "Add projects section with dynamic project loading"
- Documented design tokens and UI kits for reuse across the site.
  - `README.md`: "Add design system assets and documentation - colors, typography, components, and UI kits"


### eye-scroll-extension

Eye Scroll is a Chrome extension that lets users scroll webpages by looking down the screen using webcam-based eye tracking.

- Provides eye‑controlled scrolling via webcam.
  - `README.md`: "A Chrome extension that allows you to scroll webpages by looking down the screen using your webcam and eye tracking technology."
- Offers a slider to adjust scroll speed from 1 to 10 pixels per scroll.
  - `README.md`: "Adjustable Speed: Control scroll speed with a slider (1-10 pixels per scroll)"
- Includes a toggle button to enable or disable eye tracking.
  - `README.md`: "Manual Activation: Enable/disable tracking with a simple toggle button"
- Shows a tiny scroll indicator when scrolling is active.
  - `README.md`: "Visual Feedback: Tiny scroll indicator appears when actively scrolling"
- Processes all data locally with no server communication.
  - `README.md`: "Privacy-Focused: All processing happens locally - no data is sent to servers"
- Targets 60 fps smooth scrolling while using 20‑30% CPU during active tracking.
  - `README.md`: "Target: 60fps smooth scrolling - CPU Usage: 20-30% during active tracking"


### websocket_mcqs1

A MERN‑stack application that lets users create multiple‑choice questions and play against each other with real‑time score updates.

- The repository implements a real‑time MCQ battle.
  - `README.md`: "Realtime MCQs battle"
- The application is built with the MERN stack.
  - `README.md`: "It is written in Javascript using MERN stack"
- Installation requires running `npm install`.
  - `README.md`: "npm install"
- The server is started with `node server.js`.
  - `README.md`: "node server.js"
- Scores are observed in real time by all participants.
  - `README.md`: "observe their score in realtime"
- Users can create MCQs and play against each other.
  - `README.md`: "create MCQs and play against each other"


### Rule-BasedCustomKafkaPartitioner

A Java 21 demo that implements a custom Kafka partitioner routing messages based on key rules.

- Implements a custom Kafka partitioner that routes messages based on key rules.
  - `README.md`: "This project demonstrates a custom Kafka partitioner that routes messages based on rules: `USA` keys go to the last partition, others are hashed across the rema"
- The partitioner sends keys equal to "USA" to the last partition and hashes other keys to the remaining partitions.
  - `README.md`: "Custom partitioner logic: `USA` → last partition; others → hash % (numPartitions - 1)."
- A producer demo sends sample messages with keys including "USA" to the topic "orders".
  - `README.md`: "The producer sends sample messages with keys `INDIA`, `GERMANY`, `USA`, `FRANCE`, `USA` to topic `orders`."
- Producer output shows "USA" messages are routed to partition 3, the last partition.
  - `README.md`: "Sent key=INDIA value=Order-1 → partition=1 Sent key=GERMANY value=Order-2 → partition=1 Sent key=FRANCE value=Order-4 → partition=2 Sent key=USA value=Order-3 →"
- The demo topic is created with four partitions, enabling the rule‑based routing.
  - `README.md`: "Topic `orders` is auto‑created with 4 partitions by default."
- Kafka is run locally using Docker Compose for a single‑node setup.
  - `README.md`: "docker-compose.yml for a single‑node Kafka setup."


### tender_search_app

A web‑based tool that lets users search and filter Indian tenders using a Gradio interface.

- Provides search functionality across multiple Indian tender sources.
  - `README.md`: "Search for tenders across multiple sources."
- Allows filtering of search results by recency and relevance.
  - `README.md`: "Filter tenders based on recency and relevance."
- Implements a user‑friendly interface using Gradio.
  - `README.md`: "User-friendly interface built with Gradio."
- Builds the application stack with Python, Gradio, Langchain, BeautifulSoup, Requests, and Pandas.
  - `README.md`: "Technologies Used - Python - Gradio - Langchain - BeautifulSoup - Requests - Pandas"
- Runs the application via a Python script that starts the Gradio server.
  - `README.md`: "Run the application:    ```    python main.py    ```"


### sandbox-agent

sandbox-agent is a Python-based LangGraph agent framework that supports an educative tool, lazy loading, and web search integration.

- Credentials for the educative tool are stored in an environment variable.
  - `commits/6d8d1e9879`: "Move EDUCATIVE_COOKIES_B64 from source code to environment variable"
- An educative feature allows users to ask questions about a course using a cookie.
  - `commits/f7281212c5`: "added educative feature just we can just give cookie and ask question about the course"
- A dummy stub container is provided for local tool execution, with a clear error message when run in LangGraph Studio.
  - `commits/1a0c192bc9`: "added dummy stub container working but some other error NotImplementedError('Tool execution is not available in LangGraph Studio. Please run the agent locally ("
- The repository includes a tavily web search tool for external data retrieval.
  - `commits/5d6794c559`: "updating added  tavily web search"
- An explicit NotImplementedError informs users that tool execution requires running the agent locally.
  - `commits/1a0c192bc9`: "NotImplementedError('Tool execution is not available in LangGraph Studio. Please run the agent locally (`python harness.py`) to execute tools.')"


### crud_api1

A basic CRUD API built with Node.js and Express.js.

- The project is a basic CRUD API using Node.js and Express.js.
  - `README.md`: "Basic CRUD API in nodeJs and ExpressJs"
- A POST endpoint for creating users was added.
  - `commits/5ccd951997`: "adding USER POST"
- A GET endpoint to retrieve a user by ID was implemented.
  - `commits/662abdac04`: "creating GET /user/:id Endpoint"
- A PATCH endpoint for updating a user by ID was added.
  - `commits/e30cc0b167`: "Create the PATCH /users/:id Endpoint"
- A DELETE endpoint to remove a user by ID was implemented.
  - `commits/7c7df3e5de`: "CREATE delete /user/:id"
- The project uses nodemon to automatically restart the server on code changes.
  - `commits/359f8e20c3`: "added start as nodemon in package.json"


### simpleChat

A Go‑based chat API that forwards user questions to the GROQ language model and returns responses.

- The application requires a GROQ API key and a port number supplied via a .env file.
  - `README.md`: "GROQ_API_KEY=your key PORT=8080"
- The API server is started by running `go run main.go`.
  - `README.md`: "go run main.go"
- Postman is used to test the POST endpoint at http://localhost:8080/api/questions.
  - `README.md`: "open postman and hit a post request at the endpoint http://localhost:8080/api/questions"


## Run cost

- Groq: 4025 tokens across 1 calls
- Cache hit rate: 0%
- Wall clock: 6s
