Curso : Cómputo Cientifico para Probabilidad y Estadística
Título : Tarea 1
Nombre : Francisco Valente Castro

El código para la tarea 1 consiste de 4 archivos con extensión ".py" que son :

- substitution.py  : Implementa los algoritmos de backward y forward substitution. (Ejercicio 1)
- factorization.py : Implementa los algoritmos de factorización LUP, LU y Cholesky. (Ejercicios 2,3,5)
- solve_system.py  : Implementa la solución de un sistema de ecuaciones. (Ejercicio 4)
- compare.py : Implementa la comparación de tiempos de ejecución entre LUP y Cholesky. (Ejercicio 6)

Algunos de los archivos ".py" reciben parámetros opcionales para modificar la ejecución "python archivo.py".
Estos parámetros son :

- python factorization.py semilla : El parámetro semilla cambia el valor 0 por defecto. Cambia los números generados aleatoriamente.
- python solve_system.py semilla : El parámetro semilla cambia el valor 0 por defecto. Cambia los números generados aleatoriamente.
- compare.py tamaño_maximo_de_matriz tamaño_fijo_de_matriz tamaño_de_muestra_para_histograma mostrar_información_de_debug
             - El parámetro más importante es el tamaño de la matriz más grande que generaremos para comparar las ejecuciones.
             - El segundo y tercer parámetro sirven para comparar los tiempos de ejecución con una muestra de matrices de tamaño fijo. 
             - El último parámetro puede ser "True" ó "False" y ocasiona que se imprima más información de ejecución del código.

Nota : Los resultados presentados en el reporte se hicieron con la semilla para "np.random.seed" igual a cero. 