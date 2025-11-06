import ollama

def explica_prediccion(entrada, prediccion):
    prompt = f"El vino con atributos: {entrada} tiene una calidad predicha de {prediccion}. Explica brevemente por qué según tendencia de datos químicos."
    respuesta = ollama.generate(
        model="llama2",
        prompt=prompt
    )
    return respuesta['response']
