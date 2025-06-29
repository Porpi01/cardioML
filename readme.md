
Este proyecto implementa un sistema de Inteligencia Artificial para predecir el riesgo de Enfermedad Cardiovascular (ECV) en pacientes. Su objetivo principal es maximizar la detección de casos reales de ECV, permitiendo intervenciones tempranas y optimizando la atención sanitaria.

Nuestro sistema busca identificar la mayor cantidad posible de pacientes con ECV para asegurar una intervención temprana y evitar consecuencias graves. Priorizamos la detección exhaustiva (Recall), ya que no detectar a un paciente enfermo es el mayor riesgo en el ámbito médico.

Se utilizó el "Cardiovascular Disease Dataset" de Kaggle, con 70.000 registros de pacientes. Contiene información clínica básica y hábitos de vida (edad, tensión, colesterol, etc.), similar a la obtenida en una consulta médica estándar y con un buen equilibrio entre casos positivos y negativos.

Requisitos y Librerías Necesarias
Para ejecutar y trabajar con este proyecto, necesitarás las siguientes librerías de Python:
pandas (para manejo de datos)
numpy (para operaciones numéricas)
seaborn (para visualización de datos)
matplotlib (para visualización de datos)
scikit-learn (para modelos de Machine Learning y preprocesamiento)
xgboost (si se considera ese modelo específico para comparación o uso)
Puedes instalarlas usando pip:
pip install pandas numpy seaborn matplotlib scikit-learn xgboost