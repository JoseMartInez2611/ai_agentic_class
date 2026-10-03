# Week 6 Notes — ADK vs. LangGraph tool-calling

Third tool added to both agents: `recommend_next_topics(completed_weeks: list[int]) -> dict`,
which returns the next uncompleted week and its topic given the weeks a student has already
finished. Chains naturally with `estimate_study_hours` (and, in one test prompt, also with
`get_course_topic`).

## Behavioral difference observed

On the same prompt requiring 2+ chained tool calls, the **ADK** agent's final answer comes back
as a clean plain string (`event.content.parts[0].text`) — ready to print as-is. The **LangGraph**
agent's final answer (`result["messages"][-1].content`, backed by `ChatGoogleGenerativeAI`) comes
back wrapped in a block-list structure instead:

```
[{'type': 'text', 'text': '...', 'extras': {'signature': '...'}}]
```

The actual generated text is identical in substance between the two frameworks — only the
wrapper format differs, and any code consuming the LangGraph agent's output has to unwrap that
list/dict to get the plain string. This is the same wrapper-format difference already observed
between the raw Gemini SDK and LangChain back in Week 1/Week 2 (`response.text` vs. LangChain's
block-list `.content`); it's not specific to agents, it's just how `ChatGoogleGenerativeAI`
represents a response either way.

Secondary observation: for the test prompt that needs `recommend_next_topics` followed by two
independent follow-ups (`get_course_topic` and `estimate_study_hours`, neither depending on the
other's result), **both** frameworks issued those two tool calls in the same reasoning step
rather than strictly one at a time — i.e. both parallelize independent tool calls instead of
forcing a fully serial Thought→Action→Observation loop for every single call.

## A third finding, from actually trying to break a tool

Testing the activity's suggested break ("rename a tool without updating the call site") turned
up something more interesting than expected, in two parts:

1. Renaming the function but leaving the *old* name in the system instruction text does **not**
   break anything — the model only ever sees the real, current tool name/schema (introspected
   from the function itself), so stale prose in the instruction is just dead text, never a
   runtime error.
2. The actual break only shows up with a real Python-level bug *inside* the tool body (e.g. a
   variable renamed in one place but not another, raising `NameError` when the tool runs). In
   that case, **both ADK and LangGraph crash the entire run** with the raw Python exception,
   instead of catching it and feeding an error observation back to the model to self-correct.
   Neither framework does that translation for you automatically — a tool has to catch its own
   exceptions and return a structured `{"error": ...}` dict if graceful self-correction is the
   goal.
