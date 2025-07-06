# core/intelligents.py
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from core.mind import build_chain, kabo_state, update_mood, update_topic, update_time_and_season
from core.memory import save_memory, summarize_messages, save_episode, search_vectorstore
from core.speak import KaboTTS, play_audio
from pathlib import Path


class SimpleMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
        self._save()

    def add_user_message(self, content):
        self.add_message(HumanMessage(content=content))

    def add_ai_message(self, content):
        self.add_message(AIMessage(content=content))

    def clear(self):
        self.messages = []
        self._save()

    def get_messages(self):
        return self.messages

    def _save(self):
        save_memory(self.messages)


class KaboAI:
    def __init__(self):
        self.memory = SimpleMemory()
        self.chain = build_chain(kabo_state)

    def update_state(self, user_input):
        update_time_and_season()
        update_mood(user_input)
        update_topic(user_input)

    def get_response(self, user_input):
        self.update_state(user_input)

        history = self.memory.get_messages()
        self.memory.add_user_message(user_input)

        retrieved_facts = search_vectorstore(user_input, k=3)
        context_info = "\n".join(retrieved_facts)
        if context_info:
            context_info = f"[Relevant Facts from Memory]\n{context_info}\n"

        try:
            result = self.chain.invoke({
                "chat_history": history,
                "input": user_input,
                "context": context_info,
                **kabo_state
            })
        except Exception as e:
            print(f"Fehler beim LLM-Aufruf: {e}")
            return "Da ist etwas schiefgelaufen."

        self.memory.add_ai_message(result)

        try:
            tts = KaboTTS(audio_prompt_path=Path("models/Kikuri_VA_Sample.wav"))
            tts.speak(result)
            play_audio(Path("core/output.wav"))
        except Exception as e:
            print(f"TTS-Fehler: {e}")

        recent_messages: list[BaseMessage] = self.memory.get_messages()[-10:]
        if recent_messages:
            topic = kabo_state.get("topic", "something interesting")
            summary = summarize_messages(recent_messages)
            save_episode(f"Conversation about {topic}", summary, recent_messages)

        return result

