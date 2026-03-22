from app.nlp.faq_service import FAQService


faq_service = FAQService()


def recognize_intent(user_input: str) -> str:
    return faq_service.answer(user_input)
