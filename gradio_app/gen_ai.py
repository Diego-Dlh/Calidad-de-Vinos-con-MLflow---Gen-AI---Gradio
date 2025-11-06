import ollama

def explica_prediccion(entrada, prediccion):
    entrada_str = ", ".join([str(x) for x in entrada])
    prompt = f"El vino con atributos: {entrada_str} tiene una calidad predicha de {prediccion:.2f}. Explica brevemente por qué según tendencia de datos químicos."
    respuesta = ollama.generate(
        model="llama2",
        prompt=prompt
    )
    return respuesta['response']
