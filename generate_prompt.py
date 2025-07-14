from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Type definitions
@dataclass
class AskFlowItem:
    field: str
    priority: str  # 'high' | 'medium' | 'low'
    preferred_stage: str  # 'before_escalation' | 'after_troubleshooting'
    reason: str

@dataclass
class EscalationPolicy:
    threshold_fail_steps: int

@dataclass
class OEMConfig:
    product_id: str
    oem: str
    ask_flow: List[AskFlowItem]
    escalation_policy: EscalationPolicy
    tools_allowed: List[str]
    voice_tone: str  # 'friendly' | 'formal'
    language: str

@dataclass
class SessionState:
    step_count: int
    fields_collected: Dict[str, str]  # e.g., { 'serial_number': '1234', 'location': '' }

@dataclass
class RAGChunk:
    content: str

@dataclass
class ChatCompletionMessage:
    role: str  # 'system' | 'user' | 'assistant'
    content: str

# Helper to format OEMConfig into a readable string
def format_oem_config(oem_config: OEMConfig) -> str:
    ask_flow_lines = [
        f"- Ask for {item.field} ({item.priority}) {item.preferred_stage}: {item.reason}"
        for item in oem_config.ask_flow
    ]
    tools_line = f"- Use tools: {', '.join(oem_config.tools_allowed)}" if oem_config.tools_allowed else None
    escalation_line = f"- Escalate after {oem_config.escalation_policy.threshold_fail_steps} failed suggestions"
    tone_line = f"- Speak in {oem_config.voice_tone} tone ({oem_config.language})"
    lines = [
        f"OEM: {oem_config.oem} | Product ID: {oem_config.product_id}",
        *ask_flow_lines,
        tools_line,
        escalation_line,
        tone_line,
    ]
    return '\n'.join([line for line in lines if line])

# Helper to format session state
def format_session_state(session_state: SessionState, ask_flow_fields: List[str]) -> str:
    field_lines = [
        f"- {field} = {session_state.fields_collected.get(field).strip() if session_state.fields_collected.get(field, '').strip() else 'not yet provided'}"
        for field in ask_flow_fields
    ]
    return '\n'.join([
        "Session so far:",
        f"- step_count = {session_state.step_count}",
        *field_lines
    ])

# Helper to format RAG chunks
def format_rag_chunks(rag_chunks: Optional[List[RAGChunk]]) -> List[ChatCompletionMessage]:
    if not rag_chunks:
        return []
    return [
        ChatCompletionMessage(role="system", content=f"Manual says: {chunk.content}")
        for chunk in rag_chunks
    ]

# Main function
def generate_prompt(
    user_input: str,
    base_prompt: str,
    oem_config: OEMConfig,
    session_state: SessionState,
    rag_chunks: Optional[List[RAGChunk]] = None
) -> List[Dict[str, str]]:
    messages: List[ChatCompletionMessage] = [
        ChatCompletionMessage(role="system", content=base_prompt),
        ChatCompletionMessage(role="system", content=format_oem_config(oem_config)),
        ChatCompletionMessage(role="system", content=format_session_state(session_state, [item.field for item in oem_config.ask_flow])),
        *format_rag_chunks(rag_chunks),
        ChatCompletionMessage(role="user", content=user_input),
    ]
    # Convert dataclass objects to dicts for OpenAI API
    return [msg.__dict__ for msg in messages]

# Example usage (comment out in production)
"""
if __name__ == "__main__":
    oem_config = OEMConfig(
        product_id="samsung_eco_7kg",
        oem="Samsung",
        ask_flow=[
            AskFlowItem(field="serial_number", priority="high", preferred_stage="after_troubleshooting", reason="Needed for support ticket"),
            AskFlowItem(field="location", priority="medium", preferred_stage="before_escalation", reason="Helps locate service centers"),
        ],
        escalation_policy=EscalationPolicy(threshold_fail_steps=2),
        tools_allowed=["raise_ticket", "connect_support"],
        voice_tone="friendly",
        language="English"
    )
    session_state = SessionState(
        step_count=2,
        fields_collected={"serial_number": "", "location": "Delhi"}
    )
    rag_chunks = [
        RAGChunk(content="Ensure the drain pipe is not kinked or blocked."),
        RAGChunk(content="Check that the door is fully latched and the rubber seal is intact."),
    ]
    prompt = generate_prompt(
        user_input="My washing machine is leaking water.",
        base_prompt="You are a helpful assistant that helps users troubleshoot appliances by speaking naturally and guiding them clearly...",
        oem_config=oem_config,
        session_state=session_state,
        rag_chunks=rag_chunks
    )
    from pprint import pprint
    pprint(prompt)
""" 