# Vera Merchant AI Assistant — magicpin AI Challenge

## Approach & Architecture

Vera is designed around a **4-Context Deterministic Synthesis Engine** backed by a **Multi-Turn Intent Router**. Every message composed by Vera — whether merchant-facing (`send_as: "vera"`) or customer-facing on behalf of the merchant (`send_as: "merchant_on_behalf"`) — is evaluated across four structured context layers:

```
                          ┌───────────────────────────┐
   CategoryContext   ───► │                           │
   MerchantContext   ───► │   Deterministic 4-Context │───► ComposedMessage {body, cta,
   TriggerContext    ───► │   Synthesis Engine        │     send_as, suppression_key,
   CustomerContext?  ───► │                           │     rationale}
                          └───────────────────────────┘
```

### 1. Specificity & Grounding (Zero Hallucination)
- **Concrete Anchors**: Messages extract exact numbers from the context (`views`, `calls`, `ctr`, `delta_7d`, `lapsed_count`).
- **Catalog Pricing**: Offers strictly reference the merchant's active offers (e.g., `Dental Cleaning @ ₹299`, `Haircut @ ₹99`) or vertical catalog defaults. No generic "10% off".
- **Source Citations**: Research and regulatory triggers embed exact citations (e.g., `JIDA Oct 2026, p.14`, `DCI circular 2026-11-04`).
- **No URLs**: Strictly enforces WhatsApp/Meta delivery rules (zero URL injection).

### 2. Category Voice Adaptation
- **Dentists**: Clinical-peer tone (`Dr. <LastName>`), technical terminology (`caries recurrence`, `fluoride varnish`), legal taboos enforced (no `"cure"`, `"guaranteed"`).
- **Salons**: Warm, friendly, visual style (`hair spa @ ₹499`, `bridal prep`).
- **Restaurants**: Operator-pragmatic (`Match Night Combo`, `DC vs MI at Arun Jaitley Stadium`).
- **Gyms**: Motivational coaching (`renewals`, `kids yoga summer camp`).
- **Pharmacies**: Trustworthy, compliant, precise (`atorvastatin recall`, `chronic refill due`).

### 3. Multi-Turn & Edge-Case Handling
- **Auto-Reply Detection**: Detects WhatsApp Business canned responses (`"Thank you for contacting..."`) and backs off (`action: "wait"`) or exits (`action: "end"` after repeated turns) without wasting quota.
- **Intent Transition**: When a merchant says *"Yes, let's do it"*, Vera switches immediately from pitch/qualification mode to action execution (`action: "send"`, `"Drafted... reply CONFIRM"`).
- **Hostile / Opt-Out**: Gracefully handles opt-outs (`"stop"`, `"not interested"`) with clean session termination (`action: "end"`).
- **Off-Topic Deflection**: Gently declines out-of-scope requests (e.g., GST filing) while steering back to growth goals.

---

## Performance & Deliverables

- `bot.py`: Complete Python module & FastAPI HTTP server exposing `/v1/healthz`, `/v1/metadata`, `/v1/context`, `/v1/tick`, and `/v1/reply`.
- `submission.jsonl`: Pre-generated 30 canonical test pair compositions (`T01`–`T30`).
- `conversation_handlers.py`: Multi-turn response handler module.
- `test_bot.py`: Comprehensive pytest integration test suite.

---

## Trade-offs Made & Future Context

1. **Deterministic Speed vs LLM Dynamic Variance**: We prioritized a zero-latency (<50ms), deterministic rule-synthesis engine over unbounded LLM calls to guarantee 100% compliance with WhatsApp guidelines, exact number retention, and strict budget caps.
2. **Additional Context That Would Help Most**: Direct real-time clinic/salon SaaS integration (e.g., Practo, Dentcubate) for live slot availability, and past customer conversation transcripts to further refine Hinglish code-mix preferences.
