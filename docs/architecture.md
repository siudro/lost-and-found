# Architecture

How the pieces connect, in two views: what runs when a video arrives, and what runs when a question arrives.

## Component diagram

A UML component diagram. Boxes are modules; arrows are "calls" or "reads/writes".

```mermaid
graph TB
    subgraph Client
        Browser["Browser<br/>index.html"]
        Curl["curl / any HTTP client"]
    end

    subgraph "FastAPI process"
        Main["app/main.py<br/>FastAPI app<br/>startup diagnostics"]
        API["/api/chat · /api/ingest<br/>/api/health · /api/providers"]
    end

    subgraph "Pipeline — when video arrives"
        Src["pipeline.SourceSpec<br/>video source + clock"]
        Run["pipeline.run_pipeline"]
        FF["ffmpeg / ffprobe<br/>decode · sample · probe"]
        AW["pipeline.analyse_window<br/>motion stability → TrackSamples"]
        Inf["unattended.infer<br/>state machine"]
    end

    subgraph "Domain — pure logic, no I/O"
        TU["timeutil<br/>venue clock · windows · offsets"]
        QP["query.parse<br/>item · colour · date"]
        Ans["answer.compose<br/>honesty rules"]
    end

    subgraph "Persistence"
        Store["store.Store<br/>SQLite"]
        DB[("foundit.db<br/>sources · windows<br/>observations · events<br/>evidence · handovers")]
    end

    subgraph "AI providers — all optional"
        Det["rule_based_detector<br/>no key"]
        SAM["sam_segmenter<br/>gated checkpoint"]
        Cap["captioner<br/>API key"]
        LLM["llm_answerer<br/>API key"]
    end

    Browser -->|HTTP| API
    Curl -->|HTTP| API
    API --> Main
    Main -->|runs at startup| Store

    API -->|"POST /api/ingest"| Run
    Run --> Src
    Run --> FF
    FF -->|frames| AW
    AW --> Inf
    Inf --> TU
    AW --> TU
    AW -.->|"optional"| Det
    AW -.->|"optional"| SAM
    AW -.->|"optional"| Cap

    Run -->|WindowResult| Store
    Store --> DB

    API -->|"POST /api/chat"| QP
    QP --> TU
    QP --> Store
    Store -->|EventRows| Ans
    Ans -->|Answer| API
    Ans -.->|"optional rephrase"| LLM
```

## Sequence — a question arrives

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI /api/chat
    participant Q as query.parse
    participant S as store.Store
    participant C as answer.compose

    U->>A: "Have you seen a gray jacket? I lost it yesterday."
    A->>Q: parse(text, now, Asia/Riyadh)
    Q->>Q: expand synonyms → jacket, gray
    Q->>Q: "yesterday" → 19 Nov 21:00Z … 20 Nov 21:00Z
    Q-->>A: ParsedQuery
    A->>S: coverage_between(range)
    S-->>A: windows + camera gaps
    A->>S: search_events(range, item, colour, ¬worn)
    S-->>A: EventRows
    A->>C: compose_matches(rows, limitations, handover_state)
    Note over C: handover_state == no_intake_records<br/>→ append "cannot be verified"
    C-->>A: Answer
    A-->>U: matches + evidence + coverage limits
```

## Sequence — a video arrives

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI /api/ingest
    participant P as pipeline
    participant F as ffmpeg
    participant I as unattended.infer
    participant S as store.Store

    U->>A: upload clip + recorded_at
    A->>P: run_pipeline(SourceSpec)
    P->>F: probe duration
    P->>F: sample frames (1 fps, 320×180)
    loop each 10-minute window
        P->>P: analyse_window → TrackSamples
        P->>I: infer(samples, is_worn, present_at_start)
        I-->>P: state, onset_certainty, rules_fired
    end
    P-->>A: [WindowResult]
    A->>S: upsert source, windows, observations, events, evidence
    A-->>U: window count + processing log
```

## Class relationships

```mermaid
classDiagram
    class SourceSpec {
        +source_id
        +camera_id
        +kind
        +clock
        +recording_started_at
        +duration_seconds
    }
    class WindowResult {
        +window
        +observations
        +events
        +evidence
    }
    class TrackSample {
        +offset_s
        +position
        +nearest_person_distance_ratio
        +visible
    }
    class Inference {
        +state
        +onset_certainty
        +seconds_stationary
        +seconds_deserted
        +rules_fired
    }
    class ParsedQuery {
        +item_terms
        +colours
        +date_range
        +assumed_date
    }
    class Answer {
        +answer_mode
        +body
        +matches
        +limitations
        +handover_status
    }
    class Match {
        +event_id
        +state_claim
        +local_time
        +evidence_available
    }

    SourceSpec --> WindowResult : produces
    WindowResult "1" --> "*" TrackSample : analysed from
    TrackSample --> Inference : fed to infer()
    WindowResult --> Inference : carries verdict
    ParsedQuery --> Answer : scopes
    Answer "1" --> "*" Match
```

## The three modules that must not import anything

`timeutil`, `query`, `answer` and `unattended` are pure: no network, no disk, no model. That is deliberate — they are the parts whose correctness the submission depends on, and purity is what makes them testable without mocking anything.

Everything that touches the outside world (ffmpeg, SQLite, API keys) lives in `pipeline`, `store`, and `main`.

## Where the boundary is

```mermaid
graph LR
    subgraph "Pure — no I/O"
        TU2[timeutil]
        UN2[unattended]
        QP2[query]
        AN2[answer]
    end
    subgraph "Impure — has side effects"
        PI2[pipeline]
        ST2[store]
        MA2[main]
        PV2[ai.providers]
    end
    PI2 --> TU2
    PI2 --> UN2
    ST2 --> TU2
    MA2 --> PI2
    MA2 --> ST2
    ST2 --> AN2
    MA2 --> QP2
    PV2 -.->|"declares what AI is needed"| MA2
```

Swapping the motion detector for SAM means changing `pipeline.analyse_window` only. Nothing in `store`, `answer`, `query` or `timeutil` needs to know.
