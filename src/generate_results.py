import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
csv_path = 'data/resultados_analisis.csv'
df = pd.read_csv(csv_path)

# Filtrar "No_Detectado" para la gráfica de evolución, pero mantenerlo para el total
df_filtered = df[df['emocion'] != 'No_Detectado']

# Configurar el estilo visual
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 2. Gráfica 1: Evolución de las Emociones en el Tiempo
# Asignamos un valor numérico a las emociones para poder graficarlas
emociones_lista = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
emocion_to_num = {em: i for i, em in enumerate(emociones_lista)}
df_filtered['emocion_num'] = df_filtered['emocion'].map(emocion_to_num)

sns.lineplot(x='segundo', y='emocion_num', data=df_filtered, ax=ax1, marker='o', color='b')
ax1.set_yticks(range(len(emociones_lista)))
ax1.set_yticklabels(emociones_lista)
ax1.set_title('Evolución Emocional durante la Sesión')
ax1.set_xlabel('Tiempo (segundos)')
ax1.set_ylabel('Emoción Detectada')

# 3. Gráfica 2: Distribución porcentual (Gráfica de Pastel)
distribucion = df['emocion'].value_counts()
distribucion.plot.pie(autopct='%1.1f%%', ax=ax2, startangle=140, cmap='Set3')
ax2.set_title('Distribución Total de Estados Emocionales')
ax2.set_ylabel('') # Quitar etiqueta lateral

plt.tight_layout()

# 4. Guardar la imagen para el documento de tesis
plt.savefig('data/grafica_resultados.png')
print("Gráficas generadas exitosamente en 'data/grafica_resultados.png'")
plt.show()