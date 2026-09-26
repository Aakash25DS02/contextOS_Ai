import os
import sqlite3
import re
import html
import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

load_dotenv(override=True)

# Local development: read from .env.
# Streamlit Cloud: read from st.secrets.
env_groq_key = os.getenv("GROQ_API_KEY", "").strip().strip("'").strip('"')

try:
    secret_groq_key = str(
        st.secrets.get("GROQ_API_KEY", "")
    ).strip().strip("'").strip('"')
except Exception:
    secret_groq_key = ""

GROQ_API_KEY = env_groq_key or secret_groq_key

DATABASE = "contextos.db"


# ============================================================
# DATABASE
# ============================================================

def create_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person TEXT,
            action TEXT,
            object TEXT,
            deadline TEXT,
            reason TEXT,
            original_text TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            current_activity TEXT,
            memory_used TEXT,
            prediction TEXT,
            confidence TEXT,
            decision TEXT
        )
    """)

    connection.commit()
    connection.close()


create_database()


# ============================================================
# MEMORY
# ============================================================

def get_memories():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            person,
            action,
            object,
            deadline,
            reason,
            original_text
        FROM memories
        ORDER BY id DESC
    """)

    memories = cursor.fetchall()

    connection.close()

    return memories


def save_memory(
    person,
    action,
    object_name,
    deadline,
    reason,
    original_text
):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO memories
        (
            person,
            action,
            object,
            deadline,
            reason,
            original_text
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        person,
        action,
        object_name,
        deadline,
        reason,
        original_text
    ))

    connection.commit()
    connection.close()


def delete_memory(memory_id):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM memories WHERE id = ?",
        (memory_id,)
    )

    connection.commit()
    connection.close()


# ============================================================
# CONTEXT HISTORY
# ============================================================

def save_context_event(
    current_activity,
    memory_used,
    prediction,
    confidence,
    decision
):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO context_events
        (
            current_activity,
            memory_used,
            prediction,
            confidence,
            decision
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        current_activity,
        memory_used,
        prediction,
        confidence,
        decision
    ))

    connection.commit()
    connection.close()


def get_context_events():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            current_activity,
            memory_used,
            prediction,
            confidence,
            decision
        FROM context_events
        ORDER BY id DESC
    """)

    events = cursor.fetchall()

    connection.close()

    return events


def delete_context_event(event_id):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM context_events WHERE id = ?",
        (event_id,)
    )

    connection.commit()
    connection.close()


# ============================================================
# DUPLICATE MEMORY DETECTION
# ============================================================

def find_duplicate_memories(user_input):

    memories = get_memories()

    if not memories:
        return []

    memory_text = ""

    for memory in memories:

        memory_text += f"""
MEMORY ID: {memory[0]}

Person: {memory[1]}
Action: {memory[2]}
Object: {memory[3]}
Deadline: {memory[4]}
Reason: {memory[5]}
Original statement: {memory[6]}

"""

    result = ask_groq([

        {
            "role": "system",

            "content": """
You are the ContextOS Memory Management Engine.

Determine whether the NEW memory is substantially
the same as any existing memory.

Consider meaning, person, object, action, deadline,
and reason.

Two memories should be considered duplicates when
they describe essentially the same situation or intent,
even if the wording is different.

Do not mark memories as duplicates merely because
they share one word.

Return ONLY:

DUPLICATE: <memory id>

or

NONE
"""
        },

        {
            "role": "user",

            "content": f"""
NEW MEMORY:

{user_input}

EXISTING MEMORIES:

{memory_text}

Is the new memory a duplicate?
"""
        }

    ])

    result = result.strip().upper()

    if "NONE" in result:
        return []

    match = re.search(
        r"DUPLICATE:\s*(\d+)",
        result
    )

    if not match:
        return []

    duplicate_id = int(match.group(1))

    for memory in memories:

        if memory[0] == duplicate_id:
            return [memory]

    return []


# ============================================================
# GROQ
# ============================================================

def ask_groq(messages):

    client = Groq(
        api_key=GROQ_API_KEY
    )

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=messages,

        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(user_input):

    result = ask_groq([

        {
            "role": "system",

            "content": """
You are the ContextOS Intent Engine.

Classify the user's message into exactly ONE:

STORE
RECALL
PREDICT

STORE:
The user gives new information that should be remembered.

RECALL:
The user asks about something in memory.

PREDICT:
The user describes a current action and ContextOS
should use previous context to predict what they may need.

Examples:

"Rahul needs my presentation tomorrow."
STORE

"What does Rahul need?"
RECALL

"I'm opening Google Drive."
PREDICT

Return ONLY one word.
"""
        },

        {
            "role": "user",

            "content": user_input
        }

    ])

    result = result.strip().upper()

    if "STORE" in result:
        return "STORE"

    if "RECALL" in result:
        return "RECALL"

    if "PREDICT" in result:
        return "PREDICT"

    return "RECALL"


# ============================================================
# CONTEXT EXTRACTION
# ============================================================

def extract_context(user_input):

    return ask_groq([

        {
            "role": "system",

            "content": """
You are the ContextOS Context Engine.

Extract:

Person
Action
Object
Deadline
Reason

Infer information only when strongly supported.

Never invent information.

Return exactly:

Person: ...
Action: ...
Object: ...
Deadline: ...
Reason: ...

Use "Not specified" when unknown.
"""
        },

        {
            "role": "user",

            "content": user_input
        }

    ])


# ============================================================
# PARSE CONTEXT
# ============================================================

def parse_context(context):

    person = "Not specified"
    action = "Not specified"
    object_name = "Not specified"
    deadline = "Not specified"
    reason = "Not specified"

    for line in context.split("\n"):

        line = line.strip()

        if line.lower().startswith("person:"):

            person = line.split(
                ":",
                1
            )[1].strip()

        elif line.lower().startswith("action:"):

            action = line.split(
                ":",
                1
            )[1].strip()

        elif line.lower().startswith("object:"):

            object_name = line.split(
                ":",
                1
            )[1].strip()

        elif line.lower().startswith("deadline:"):

            deadline = line.split(
                ":",
                1
            )[1].strip()

        elif line.lower().startswith("reason:"):

            reason = line.split(
                ":",
                1
            )[1].strip()

    return (
        person,
        action,
        object_name,
        deadline,
        reason
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(user_input):

    memories = get_memories()

    if not memories:
        return []

    memory_text = ""

    for memory in memories:

        memory_text += f"""
MEMORY ID: {memory[0]}

Person: {memory[1]}
Action: {memory[2]}
Object: {memory[3]}
Deadline: {memory[4]}
Reason: {memory[5]}
Original statement: {memory[6]}

"""

    result = ask_groq([

        {
            "role": "system",

            "content": """
You are the ContextOS Semantic Memory Engine.

Find memories that are meaningfully related to
the current user message.

Understand meaning, intent and situation.

Do NOT rely only on matching words.

Only select memories with a reasonable contextual
connection.

Return ONLY memory IDs separated by commas.

Maximum 3 IDs.

If there is no meaningful connection:

NONE
"""
        },

        {
            "role": "user",

            "content": f"""
CURRENT USER MESSAGE:

{user_input}

STORED MEMORIES:

{memory_text}

Find relevant memory IDs.
"""
        }

    ])

    result = result.strip()

    if result.upper() == "NONE":
        return []

    relevant = []

    for item in result.split(","):

        try:

            memory_id = int(
                item.strip()
            )

            for memory in memories:

                if memory[0] == memory_id:

                    relevant.append(memory)

        except ValueError:

            continue

    return relevant


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict(
    user_input,
    relevant_memories
):

    memory_context = ""

    for memory in relevant_memories:

        memory_context += f"""
Memory #{memory[0]}

Person: {memory[1]}
Action: {memory[2]}
Object: {memory[3]}
Deadline: {memory[4]}
Reason: {memory[5]}
Original: {memory[6]}

"""

    return ask_groq([

        {
            "role": "system",

            "content": """
You are the ContextOS Proactive Prediction Engine.

The user is currently doing something.

Use previous memories to predict what the user
may need or intend next.

Evaluate the strength of the contextual connection.

HIGH confidence:
The current action strongly connects to the previous
memory and there is clear evidence.

MEDIUM confidence:
There is a plausible connection but uncertainty exists.

LOW confidence:
The connection is weak or speculative.

Do not claim certainty.

Return exactly:

Likely intent: ...
Why: ...
Confidence: HIGH / MEDIUM / LOW
Helpful action: ...
"""
        },

        {
            "role": "user",

            "content": f"""
CURRENT ACTIVITY:

{user_input}

PREVIOUS CONTEXT:

{memory_context}

Predict what the user may need next.
"""
        }

    ])


# ============================================================
# CONFIDENCE
# ============================================================

def extract_confidence(prediction):

    match = re.search(
        r"Confidence:\s*(HIGH|MEDIUM|LOW)",
        prediction,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return "LOW"


# ============================================================
# DECISION ENGINE
# ============================================================

def should_interrupt(confidence):

    if confidence == "HIGH":
        return "PROACTIVE"

    elif confidence == "MEDIUM":
        return "SUGGEST"

    return "SILENT"


# ============================================================
# GRAPH HELPERS
# ============================================================

def clean_graph_text(value):

    if value is None:
        return "Not specified"

    value = str(value).strip()

    if not value:
        return "Not specified"

    return value


def graph_escape(value):

    value = clean_graph_text(value)

    value = value.replace(
        "\\",
        "\\\\"
    )

    value = value.replace(
        '"',
        '\\"'
    )

    value = value.replace(
        "\n",
        " "
    )

    return value


def create_context_graph(memories):

    if not memories:
        return None

    dot = """

digraph ContextOS {

    graph [
        rankdir=LR,
        bgcolor="transparent",
        pad="0.4",
        nodesep="0.5",
        ranksep="0.8"
    ];

    node [
        shape=box,
        style="rounded,filled",
        fontname="Arial",
        fontsize=11,
        margin="0.18,0.12"
    ];

    edge [
        fontname="Arial",
        fontsize=9
    ];

"""

    people = {}
    actions = {}
    objects = {}
    reasons = {}
    deadlines = {}

    for memory in memories:

        memory_id = memory[0]

        person = clean_graph_text(
            memory[1]
        )

        action = clean_graph_text(
            memory[2]
        )

        object_name = clean_graph_text(
            memory[3]
        )

        deadline = clean_graph_text(
            memory[4]
        )

        reason = clean_graph_text(
            memory[5]
        )

        person_id = f"person_{memory_id}"
        action_id = f"action_{memory_id}"
        object_id = f"object_{memory_id}"
        reason_id = f"reason_{memory_id}"
        deadline_id = f"deadline_{memory_id}"

        if person != "Not specified":

            people[person_id] = person

            dot += f'''
    "{person_id}" [
        label="👤 {graph_escape(person)}",
        fillcolor="lightblue"
    ];
'''

        if action != "Not specified":

            actions[action_id] = action

            dot += f'''
    "{action_id}" [
        label="⚡ {graph_escape(action)}",
        fillcolor="lightyellow"
    ];
'''

        if object_name != "Not specified":

            objects[object_id] = object_name

            dot += f'''
    "{object_id}" [
        label="📦 {graph_escape(object_name)}",
        fillcolor="lightgreen"
    ];
'''

        if reason != "Not specified":

            reasons[reason_id] = reason

            dot += f'''
    "{reason_id}" [
        label="💡 {graph_escape(reason)}",
        fillcolor="lavender"
    ];
'''

        if deadline != "Not specified":

            deadlines[deadline_id] = deadline

            dot += f'''
    "{deadline_id}" [
        label="⏰ {graph_escape(deadline)}",
        fillcolor="mistyrose"
    ];
'''

        if person != "Not specified" and action != "Not specified":

            dot += f'''
    "{person_id}" -> "{action_id}"
        [label="does"];
'''

        if action != "Not specified" and object_name != "Not specified":

            dot += f'''
    "{action_id}" -> "{object_id}"
        [label="on"];
'''

        if object_name != "Not specified" and reason != "Not specified":

            dot += f'''
    "{object_id}" -> "{reason_id}"
        [label="because"];
'''

        if object_name != "Not specified" and deadline != "Not specified":

            dot += f'''
    "{object_id}" -> "{deadline_id}"
        [label="due"];
'''

    dot += """

}
"""

    return dot


# ============================================================
# MAIN CONTEXT ANALYSIS
# ============================================================

def analyze_context(user_input):

    intent = detect_intent(
        user_input
    )

    # ========================================================
    # STORE
    # ========================================================

    if intent == "STORE":

        context = extract_context(
            user_input
        )

        (
            person,
            action,
            object_name,
            deadline,
            reason
        ) = parse_context(
            context
        )

        duplicates = find_duplicate_memories(
            user_input
        )

        if duplicates:

            return {

                "type": "DUPLICATE",

                "context": context,

                "new_data": (
                    person,
                    action,
                    object_name,
                    deadline,
                    reason,
                    user_input
                ),

                "duplicate": duplicates[0]

            }

        save_memory(
            person,
            action,
            object_name,
            deadline,
            reason,
            user_input
        )

        return {

            "type": "STORE",

            "context": context

        }

    # ========================================================
    # RECALL
    # ========================================================

    elif intent == "RECALL":

        relevant_memories = semantic_search(
            user_input
        )

        if not relevant_memories:

            return {

                "type": "RECALL",

                "memories": [],

                "answer": None

            }

        memory_context = ""

        for memory in relevant_memories:

            memory_context += f"""
Memory #{memory[0]}

Person: {memory[1]}
Action: {memory[2]}
Object: {memory[3]}
Deadline: {memory[4]}
Reason: {memory[5]}

"""

        answer = ask_groq([

            {
                "role": "system",

                "content": """
You are ContextOS.

Answer the user's question using the
retrieved memories.

Be concise and natural.

Never invent information.
"""
            },

            {
                "role": "user",

                "content": f"""
Question:

{user_input}

Relevant memories:

{memory_context}

Answer naturally.
"""
            }

        ])

        return {

            "type": "RECALL",

            "memories": relevant_memories,

            "answer": answer

        }

    # ========================================================
    # PREDICT
    # ========================================================

    else:

        relevant_memories = semantic_search(
            user_input
        )

        if not relevant_memories:

            save_context_event(

                user_input,

                "No relevant memory",

                "No prediction made",

                "LOW",

                "SILENT"

            )

            return {

                "type": "PREDICT",

                "memories": [],

                "prediction": None,

                "confidence": "LOW",

                "decision": "SILENT"

            }

        prediction = predict(
            user_input,
            relevant_memories
        )

        confidence = extract_confidence(
            prediction
        )

        decision = should_interrupt(
            confidence
        )

        memory_used = ""

        for memory in relevant_memories:

            memory_used += (

                f"Memory #{memory[0]}: "

                f"{memory[1]} → "

                f"{memory[2]} → "

                f"{memory[3]}\n"

            )

        save_context_event(

            user_input,

            memory_used,

            prediction,

            confidence,

            decision

        )

        return {

            "type": "PREDICT",

            "memories": relevant_memories,

            "prediction": prediction,

            "confidence": confidence,

            "decision": decision

        }


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="ContextOS",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.context-title {
    font-size: 42px;
    font-weight: 800;
}

.context-subtitle {
    font-size: 18px;
    opacity: 0.65;
    margin-bottom: 25px;
}

.card {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 10px;
    margin-bottom: 15px;
}

.prediction-card {
    padding: 30px;
    border-radius: 20px;
    border: 2px solid rgba(128,128,128,0.35);
    margin-top: 15px;
    margin-bottom: 20px;
}

.silent-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px dashed rgba(128,128,128,0.35);
    margin-top: 15px;
}

.history-card {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 12px;
    margin-bottom: 12px;
}

.big-text {
    font-size: 25px;
    font-weight: 700;
}

.section-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    opacity: 0.55;
}

.graph-card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 15px;
    margin-bottom: 20px;
}

.delete-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(255,80,80,0.35);
    margin-top: 15px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 ContextOS")

    st.caption(
        "Personal Context Intelligence"
    )

    st.divider()

    st.success(
        "● ContextOS Active"
    )

    st.write("### Intelligence Layer")

    st.write("🧠 Memory — Online")

    st.write("🔎 Semantic Retrieval — Online")

    st.write("🎯 Intent Detection — Online")

    st.write("🔮 Prediction — Online")

    st.write("🛡️ Attention Guard — Online")

    st.write("🧩 Context History — Online")

    st.write("🧹 Memory Management — Online")

    st.write("🕸️ Context Graph — Online")

    st.divider()

    st.write("### Attention Policy")

    st.write(
        "🟢 HIGH → Proactive"
    )

    st.write(
        "🟡 MEDIUM → Suggest"
    )

    st.write(
        "⚪ LOW → Stay Silent"
    )

    st.divider()

    st.caption(
        "ContextOS Prototype"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="context-title">🧠 ContextOS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="context-subtitle">'
    'AI that remembers <b>WHY</b> — and knows when to help.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API CHECK
# ============================================================

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY is missing from your .env file."
    )

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

memories = get_memories()

context_events = get_context_events()

people = set()

deadlines = 0


for memory in memories:

    if memory[1] != "Not specified":

        people.add(
            memory[1]
        )

    if memory[4] != "Not specified":

        deadlines += 1


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🧠 Memories",
        len(memories)
    )


with col2:

    st.metric(
        "👤 People",
        len(people)
    )


with col3:

    st.metric(
        "⏰ Deadlines",
        deadlines
    )


with col4:

    st.metric(
        "🧩 Context Events",
        len(context_events)
    )


st.divider()


# ============================================================
# PROACTIVE MODE
# ============================================================

st.markdown(
    '<div class="section-label">CONTEXT ENGINE</div>',
    unsafe_allow_html=True
)


proactive_mode = st.toggle(

    "🟢 Proactive Mode",

    value=True

)


if proactive_mode:

    st.caption(
        "ContextOS will analyze your activity "
        "and decide whether it should help."
    )

else:

    st.caption(
        "Manual Mode — ContextOS waits for your input."
    )


st.divider()


# ============================================================
# CURRENT ACTIVITY
# ============================================================

st.markdown(
    '<div class="section-label">CURRENT ACTIVITY</div>',
    unsafe_allow_html=True
)


user_input = st.text_input(

    "What are you doing or thinking?",

    placeholder="Example: I'm opening Google Drive."

)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🧠 Analyze Context",
    use_container_width=True
):

    if not user_input.strip():

        st.warning(
            "Tell ContextOS what you're doing first."
        )

    else:

        try:

            with st.spinner(
                "ContextOS is understanding your situation..."
            ):

                result = analyze_context(
                    user_input
                )

            result_type = result["type"]

            # ==================================================
            # STORE
            # ==================================================

            if result_type == "STORE":

                st.success(
                    "🧠 Context remembered."
                )

                st.subheader(
                    "Memory Created"
                )

                st.code(
                    result["context"]
                )

            # ==================================================
            # DUPLICATE
            # ==================================================

            elif result_type == "DUPLICATE":

                duplicate = result["duplicate"]

                st.warning(
                    "🔁 ContextOS found a possible duplicate memory."
                )

                st.subheader(
                    "Existing Memory"
                )

                st.markdown(
                    f"""
                    <div class="card">

                    👤 <b>WHO:</b> {html.escape(str(duplicate[1]))}

                    <br><br>

                    ⚡ <b>WHAT:</b> {html.escape(str(duplicate[3]))}

                    <br><br>

                    💡 <b>WHY:</b> {html.escape(str(duplicate[5]))}

                    <br><br>

                    ⏰ <b>WHEN:</b> {html.escape(str(duplicate[4]))}

                    <br><br>

                    📝 <b>Original:</b> {html.escape(str(duplicate[6]))}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader(
                    "New Memory"
                )

                new_data = result["new_data"]

                st.markdown(
                    f"""
                    <div class="card">

                    👤 <b>WHO:</b> {html.escape(str(new_data[0]))}

                    <br><br>

                    ⚡ <b>WHAT:</b> {html.escape(str(new_data[2]))}

                    <br><br>

                    💡 <b>WHY:</b> {html.escape(str(new_data[4]))}

                    <br><br>

                    ⏰ <b>WHEN:</b> {html.escape(str(new_data[3]))}

                    <br><br>

                    📝 <b>Original:</b> {html.escape(str(new_data[5]))}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "These memories appear to describe the same situation."
                )

                col_a, col_b = st.columns(2)

                with col_a:

                    if st.button(
                        "✅ Keep New Memory",
                        use_container_width=True,
                        key="keep_duplicate"
                    ):

                        save_memory(

                            new_data[0],
                            new_data[1],
                            new_data[2],
                            new_data[3],
                            new_data[4],
                            new_data[5]

                        )

                        st.success(
                            "🧠 New memory saved."
                        )

                        st.rerun()

                with col_b:

                    if st.button(
                        "⏭️ Skip New Memory",
                        use_container_width=True,
                        key="skip_duplicate"
                    ):

                        st.info(
                            "Existing memory kept. "
                            "No duplicate was created."
                        )

                        st.rerun()

            # ==================================================
            # RECALL
            # ==================================================

            elif result_type == "RECALL":

                relevant_memories = result["memories"]

                if not relevant_memories:

                    st.info(
                        "No relevant memory found."
                    )

                else:

                    st.subheader(
                        "🧠 ContextOS Answer"
                    )

                    st.write(
                        result["answer"]
                    )

                    with st.expander(
                        "View contextual evidence"
                    ):

                        for memory in relevant_memories:

                            st.write(
                                f"Memory #{memory[0]}"
                            )

                            st.write(
                                f"👤 {memory[1]}"
                            )

                            st.write(
                                f"⚡ {memory[2]}"
                            )

                            st.write(
                                f"📦 {memory[3]}"
                            )

                            st.write(
                                f"⏰ {memory[4]}"
                            )

                            st.write(
                                f"💡 {memory[5]}"
                            )

            # ==================================================
            # PREDICT
            # ==================================================

            elif result_type == "PREDICT":

                relevant_memories = result["memories"]

                confidence = result["confidence"]

                decision = result["decision"]

                prediction = result["prediction"]

                if decision == "SILENT":

                    st.markdown(
                        """
                        <div class="silent-card">

                        <div class="big-text">
                        🤫 ContextOS is staying silent.
                        </div>

                        <br>

                        I don't have enough contextual evidence
                        to interrupt you.

                        <br><br>

                        <b>Attention protected.</b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif decision == "PROACTIVE":

                    st.markdown(
                        """
                        <div class="prediction-card">

                        <div class="big-text">
                        🔮 ContextOS noticed something important.
                        </div>

                        <br>

                        Your current activity strongly connects
                        with something you previously intended.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(
                        prediction
                    )

                else:

                    st.markdown(
                        """
                        <div class="prediction-card">

                        <div class="big-text">
                        💡 ContextOS has a suggestion.
                        </div>

                        <br>

                        There may be a connection, but I'm
                        not completely certain.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(
                        prediction
                    )

                st.subheader(
                    "🛡️ Attention Decision"
                )

                if decision == "PROACTIVE":

                    st.success(
                        "🟢 PROACTIVE — Strong enough to help."
                    )

                elif decision == "SUGGEST":

                    st.warning(
                        "🟡 SUGGEST — Possible connection."
                    )

                else:

                    st.info(
                        "⚪ SILENT — Not enough evidence."
                    )

                st.caption(
                    f"Context confidence: {confidence}"
                )

                if relevant_memories:

                    st.subheader(
                        "🧠 Context Connection"
                    )

                    for memory in relevant_memories:

                        st.markdown(
                            f"""
                            <div class="card">

                            <b>Memory #{memory[0]}</b>

                            <br><br>

                            👤 <b>WHO:</b> {html.escape(str(memory[1]))}

                            <br>

                            ⚡ <b>WHAT:</b> {html.escape(str(memory[3]))}

                            <br>

                            💡 <b>WHY:</b> {html.escape(str(memory[5]))}

                            <br>

                            ⏰ <b>WHEN:</b> {html.escape(str(memory[4]))}

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


        except Exception as e:

            st.error(
                f"ContextOS Error: {e}"
            )


# ============================================================
# CONTEXT GRAPH
# ============================================================

st.divider()

st.header(
    "🕸️ Context Graph"
)

st.write(
    "ContextOS connects people, actions, objects, reasons "
    "and deadlines instead of storing isolated facts."
)


memories = get_memories()


if memories:

    # --------------------------------------------------------
    # GRAPH VISUALIZATION
    # --------------------------------------------------------

    graph = create_context_graph(
        memories[:10]
    )

    if graph:

        st.graphviz_chart(
            graph,
            use_container_width=True
        )

        st.caption(
            "Each connection represents contextual meaning "
            "extracted from your memories."
        )

    # --------------------------------------------------------
    # GRAPH LEGEND
    # --------------------------------------------------------

    st.markdown(
        """
        **Graph Legend**

        👤 Person → who is involved

        ⚡ Action → what is being done

        📦 Object → what the action concerns

        💡 Reason → why it matters

        ⏰ Deadline → when it matters
        """
    )

    st.divider()

    # ========================================================
    # GRAPH CONTEXT MANAGEMENT
    # ========================================================

    st.subheader(
        "🗑️ Context Graph Management"
    )

    st.caption(
        "Each graph connection is generated from a stored memory. "
        "Deleting a memory removes its connection from the graph."
    )

    for memory in memories[:10]:

        memory_id = memory[0]

        with st.expander(
            f"🕸️ Graph Context #{memory_id} — {memory[3]}"
        ):

            st.markdown(
                f"""
                <div class="graph-card">

                👤 <b>Person:</b> {html.escape(str(memory[1]))}

                <br><br>

                ⚡ <b>Action:</b> {html.escape(str(memory[2]))}

                <br><br>

                📦 <b>Object:</b> {html.escape(str(memory[3]))}

                <br><br>

                💡 <b>Reason:</b> {html.escape(str(memory[5]))}

                <br><br>

                ⏰ <b>Deadline:</b> {html.escape(str(memory[4]))}

                <br><br>

                📝 <b>Original:</b> {html.escape(str(memory[6]))}

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="delete-card">

                ⚠️ <b>Delete this graph context?</b>

                Deleting it will also delete the underlying
                memory and remove this connection from the
                Context Graph.

                </div>
                """,
                unsafe_allow_html=True
            )

            confirm_graph_delete = st.checkbox(
                "I confirm that I want to permanently delete this context.",
                key=f"confirm_graph_delete_{memory_id}"
            )

            if st.button(
                "🗑️ Delete This Graph Context",
                key=f"delete_graph_context_{memory_id}",
                use_container_width=True,
                disabled=not confirm_graph_delete
            ):

                delete_memory(
                    memory_id
                )

                st.success(
                    f"Graph Context #{memory_id} deleted."
                )

                st.rerun()

else:

    st.info(
        "Add memories first to generate your Context Graph."
    )


# ============================================================
# CONTEXT HISTORY
# ============================================================

st.divider()

st.header(
    "🧩 Context History"
)

st.write(
    "ContextOS remembers how it connected the present "
    "with the past."
)


context_events = get_context_events()


if context_events:

    for event in context_events[:10]:

        (
            event_id,
            current_activity,
            memory_used,
            prediction,
            confidence,
            decision
        ) = event

        with st.expander(
            f"🧩 Event #{event_id} — {decision}"
        ):

            st.markdown(
                f"""
                <div class="history-card">

                <div class="section-label">
                CURRENT ACTIVITY
                </div>

                <br>

                <b>{html.escape(str(current_activity))}</b>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="history-card">

                <div class="section-label">
                CONNECTED MEMORY
                </div>

                <br>

                {html.escape(str(memory_used)).replace(chr(10), "<br>")}

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="history-card">

                <div class="section-label">
                CONTEXTOS REASONING
                </div>

                <br>

                {html.escape(str(prediction)).replace(chr(10), "<br>")}

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                f"**Confidence:** {confidence}"
            )

            if decision == "PROACTIVE":

                st.success(
                    "🟢 PROACTIVE"
                )

            elif decision == "SUGGEST":

                st.warning(
                    "🟡 SUGGEST"
                )

            else:

                st.info(
                    "⚪ SILENT"
                )

            # =================================================
            # DELETE HISTORY EVENT
            # =================================================

            st.divider()

            st.markdown(
                """
                <div class="delete-card">

                ⚠️ <b>Delete this context history event?</b>

                This action permanently removes this event
                from Context History.

                </div>
                """,
                unsafe_allow_html=True
            )

            confirm_history_delete = st.checkbox(
                "I confirm that I want to permanently delete this history event.",
                key=f"confirm_history_delete_{event_id}"
            )

            if st.button(
                "🗑️ Delete This History Event",
                key=f"delete_history_event_{event_id}",
                use_container_width=True,
                disabled=not confirm_history_delete
            ):

                delete_context_event(
                    event_id
                )

                st.success(
                    f"Context History Event #{event_id} deleted."
                )

                st.rerun()

else:

    st.info(
        "No context events yet. "
        "Run a prediction to create your first event."
    )


# ============================================================
# MEMORY MANAGEMENT
# ============================================================

st.divider()

st.header(
    "🧹 Memory Management"
)

st.write(
    "Review and remove memories that ContextOS no longer needs."
)


memories = get_memories()


if memories:

    for memory in memories[:10]:

        with st.expander(
            f"🧠 Memory #{memory[0]} — {memory[3]}"
        ):

            st.write(
                f"👤 **WHO:** {memory[1]}"
            )

            st.write(
                f"⚡ **WHAT:** {memory[3]}"
            )

            st.write(
                f"💡 **WHY:** {memory[5]}"
            )

            st.write(
                f"⏰ **WHEN:** {memory[4]}"
            )

            st.write(
                f"📝 **Original:** {memory[6]}"
            )

            st.divider()

            # =================================================
            # CONFIRMATION BEFORE MEMORY DELETION
            # =================================================

            st.warning(
                "⚠️ Deleting this memory will also remove "
                "its Context Graph connection."
            )

            confirm_memory_delete = st.checkbox(
                "I confirm that I want to permanently delete this memory.",
                key=f"confirm_memory_delete_{memory[0]}"
            )

            if st.button(
                "🗑️ Delete This Memory",
                key=f"delete_memory_{memory[0]}",
                use_container_width=True,
                disabled=not confirm_memory_delete
            ):

                delete_memory(
                    memory[0]
                )

                st.success(
                    "Memory deleted successfully."
                )

                st.rerun()

else:

    st.info(
        "No memories yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ContextOS — Understand context. Remember intent. Protect attention."
)