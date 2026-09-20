# Jev

Jev is a model made by TypeSafe. Unlike a chat model, it does not write text: you send it a
question with a fixed list of possible answers, and it returns one answer with a probability.

This page explains how the optional Jev mode of this service works, what I measured, where it
breaks, and when it is worth turning on. The config fields are in the [API contract](api.md);
environment variables are in [configuration](configuration.md).

## How the mode works

1. Code reads the input and marks **candidates**. A candidate is a short piece of text with a
   start and an end position: for example a run of words that begin with a capital letter, or a
   rare word that is not a common stopword. This step is plain Python — no model, no network.
2. Every candidate is sent to Jev as one question: which of the configured labels fits this piece
   of text, or is it not an entity at all. All questions about one document travel together in a
   single request.
3. The service keeps the answers whose probability is above `min_label_probability` and drops the
   rest. Because the piece of text came from the code, the returned offsets are exact, and a
   label can never be attached to text that is not in the input.

Two things follow from the first step. Only what the code proposes can be found, so a mention
with no visible trace in the text will never reach the model. And precision is decided before
Jev is called: the model says what a piece of text is, not whether that piece was worth looking
at.

## What I measured

I first checked the first step on its own, with no model calls at all: how often do the
candidates proposed by the code contain the correct mention? On CoNLL-2003 about 99% of the
correct mentions fell inside some candidate when the code also proposed shorter and longer
versions of each candidate, and about 82% when it did not. So on text that has capital letters
the method is not limited by what it can see; the open question is how well the model labels
what it is given.

That second question depends on the text. On my test corpora F1 came out as follows:

| Text | F1 |
| --- | --- |
| English news | 0.87 |
| Russian chat messages | 0.88 |
| Technical documentation | 0.76 |
| English news with every letter lowercased | 0.67 |
| Log lines | 0.57 |

Money and speed: Jev bills only for input tokens, and one questioned candidate costs roughly
150-200 of them, so the price grows with the number of candidates rather than with the size of
the model. For short text, Jev is cheaper than a large chat model; for a long document the chat
model is cheaper, and the turning point sits near 38 tokens of text per document. A request
takes about 0.35 seconds on a warm connection and about 0.8 seconds when the connection is new.

## Problems to expect

- **The probability threshold is a per-text-type setting.** A value that worked well on news cut
  the number of found entities by 14-29% on other text types without making the remaining ones
  more accurate, and on logs and chat the threshold did not control accuracy at all. Tune it on
  your own data.
- **Jev does not recognise nonsense as nonsense.** An invented name can still receive a confident
  label. Precision is protected by the candidate step in code, not by the model.
- **Lowercase text cannot be scanned for capital letters.** The service then uses a case-free way
  of picking candidates: rare words that are not stopwords, words containing digits or dashes,
  and phrases repeated in the same document. It works, and it is only slightly less accurate on
  text that does have capital letters.
- **Never mix documents in one request.** Many questions about one document are fine; two
  documents in one state make the answers worse.
- **Do not ask about every word, and do not point at a word by its position in a long text.**
  Asking about many individual words makes the returned spans grow too wide, and in a long text
  the model loses track of which position was meant. Ask about named candidates, and keep the
  number of questions per request small.
- **A probability is not a guarantee.** It describes how the model answered one question, not how
  likely the label is to be right. Repeating the same request changes a small share of answers,
  so do not tune thresholds on differences smaller than that.
- **Span edges come from the code.** Words such as `of` and `the` must be allowed inside a
  candidate, or `New York` loses a word; and edges may need trimming, because a leading `and` can
  be captured into the span.
- **Text inside the document can steer the model.** Treat the input as untrusted data and keep the
  decision rules in code.
- **Labels must fit the text and the language.** An English news label set applied to Russian chat
  failed almost completely, while a suitable label set worked well on the same kind of text. Many
  labels also cost more and score worse: a twenty-label set needed about three times the tokens
  per question and was clearly less accurate than a small one.
- **Two published limits are optimistic.** A single request tops out near 33k tokens, not 64k, and
  the service does not enforce the advertised request rate, so add your own limits.

## When it is worth using

Use it when

- the text has capital letters and the entities you want leave a visible trace: names,
  organisations, places, identifiers, codes, formats;
- documents are short or medium, so the number of candidates stays small;
- you want exact offsets, one entity per occurrence, and no generative step that could invent a
  span;
- you want a probability per mention so uncertain cases can be sent to review.

Do not use it when

- the label leaves no visible trace in the text: obligations, findings, intentions, anything that
  can only be understood from the meaning of a whole sentence;
- documents are long, because a chat model is cheaper per document;
- the decision is security-critical, because text inside the document can influence the answer;
- the text is lowercase and the label depends on context rather than on the words themselves.

Short version: the gain is quality and control, not necessarily money. Jev gives typed answers,
exact offsets, no invented spans and a probability per mention; on long documents the chat model
stays cheaper.

## Example

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"Apple opened an office in Berlin.","config":{"labels":[{"name":"ORG","description":"Companies"},{"name":"LOCATION","description":"Cities and countries"}],"span_pipeline":{"min_label_probability":0.6}}}'
```

The response keeps the usual shape. `provider` is `typesafe`, `model` is the Jev model id, and
every entity carries exact `start` and `end` values.
