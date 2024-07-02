import random
import datetime
import sys
import time

geneSet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!. '
password = "Jack and jill went up the hill to fetch a pail of water!"


def generar_padre(longitud):
    """
    Genera una cadena aleatoria del conjunto de genes.
    """
    genes = []
    while len(genes) < longitud:
        tam_muestra = min(longitud - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, tam_muestra))
    return ''.join(genes)


def obtener_fitness(intento):
    """
    Calcula el valor de fitness, que es el número total de letras en el intento que coinciden
    con la letra en la misma posición de la contraseña.
    """
    return sum(1 for esperado, actual in zip(password, intento) if esperado == actual)


def mutar(padre):
    """
    Realiza una mutación en un gen seleccionado aleatoriamente del padre.
    """
    indice = random.randrange(0, len(padre))
    genes_hijo = list(padre)
    nuevo_gen, alterno = random.sample(geneSet, 2)
    genes_hijo[indice] = alterno if nuevo_gen == genes_hijo[indice] else nuevo_gen
    return ''.join(genes_hijo)


def mostrar(intento):
    """
    Función de visualización para monitorizar el proceso.
    """
    tiempo_transcurrido = datetime.datetime.now() - startTime
    fitness = obtener_fitness(intento)
    sys.stdout.write(f'\rGeneración: {intento} - Fitness: {fitness} - Tiempo: {tiempo_transcurrido}')
    sys.stdout.flush()
    time.sleep(.001)


if __name__ == '__main__':
    random.seed(4)
    startTime = datetime.datetime.now()
    mejor_padre = generar_padre(len(password))
    mejor_fitness = obtener_fitness(mejor_padre)
    mostrar(mejor_padre)

    while True:
        hijo = mutar(mejor_padre)
        fitness_hijo = obtener_fitness(hijo)
        mostrar(hijo)
        if mejor_fitness >= fitness_hijo:
            continue
        if fitness_hijo >= len(mejor_padre):
            break
        mejor_fitness = fitness_hijo
        mejor_padre = hijo


