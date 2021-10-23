import random
from typing import Generator

import numpy

Array = numpy.ndarray
Code = tuple[Array, Array]
TableOfSyndromes = dict[tuple[int, ...], Array]


def add(x: Array, y: Array) -> Array:
    """
    Функция вычисляет сумму двух векторов
    """
    return (x + y) % 2


def dot(x: Array, y: Array) -> Array:
    """
    Функция реализует матричное умножение
    """
    return (x @ y) % 2


def weight(x: Array) -> int:
    return sum(x)


def distance(x: Array, y: Array) -> int:
    """
    Функция вычисляет кодовое расстояние между кодовыми словами
    """
    return weight(add(x, y))


def generate_errors(n: int, number_of_errors: int) -> Generator[list[int], None, None]:
    """
    Функция возвращает генератор ошибок для заданного числа ошибок в кодовом слове
    """
    if n == 1:
        yield [0]
        if number_of_errors == 1:
            yield [1]
    elif number_of_errors == 0:
        yield [0 for _ in range(n)]
    else:
        for err in generate_errors(n - 1, number_of_errors):
            yield err + [0]
        for err in generate_errors(n - 1, number_of_errors - 1):
            yield err + [1]


def get_vectors_with_min_number_of_ones(length: int, number_of_ones: int) -> Generator[list[int], None, None]:
    """
    Функция возвращает генератор строк с заданным минимальным количеством единиц
    """
    if number_of_ones == 0:
        if length == 1:
            yield [0]
            yield [1]
        else:
            for vector in get_vectors_with_min_number_of_ones(length - 1, 0):
                yield vector + [0]
                yield vector + [1]
    elif length == number_of_ones:
        yield [1 for _ in range(length)]
    else:
        for vector in get_vectors_with_min_number_of_ones(length - 1, number_of_ones):
            yield vector + [0]
        for vector in get_vectors_with_min_number_of_ones(length - 1, number_of_ones - 1):
            yield vector + [1]


def create_table_of_syndromes(H: Array, number_of_errors: int) -> TableOfSyndromes:
    """
    Функция формирует синдромы линейного кода (n, k, d)
    """
    errors = [numpy.array(error) for error in generate_errors(H.shape[0], number_of_errors)]
    syndromes = dot(numpy.array(errors), H)
    table_of_syndromes = {}
    for syndrome, error in zip(syndromes, errors):
        table_of_syndromes[tuple(syndrome)] = error
    return table_of_syndromes


def create_code(n: int, k: int, d: int) -> Code:
    """
    Функция формирует порождающую и проверочнуюю матрицы линейного кода (n, k, d)
    """
    X = []
    suitable_vectors = [numpy.array(vector, dtype=int) for vector in
                        get_vectors_with_min_number_of_ones(n - k, max(d - 2, 2))]
    for i in range(k):
        if len(suitable_vectors) == 0:
            raise Exception('Не получается сформировать линейный код с указанными параметрами')
        vector = suitable_vectors.pop(0)
        X.append(vector)
        new_suitable_vectors = []
        for v in suitable_vectors:
            if distance(v, vector) >= d - 2:
                new_suitable_vectors.append(v)
        suitable_vectors = new_suitable_vectors
    X = numpy.array(X)
    return numpy.concatenate((numpy.eye(k, dtype=int), X), axis=1), \
           numpy.concatenate((X, numpy.eye(n - k, dtype=int)), axis=0)


def create_Hamming_code(r: int) -> Code:
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    d = 3
    return create_code(n, k, d)


def extend_Hamming_code(G: Array, H: Array) -> Code:
    H = numpy.append(H, numpy.zeros((1, H.shape[1]), dtype=int), axis=0)
    H = numpy.append(H, numpy.ones((H.shape[0], 1), dtype=int), axis=1)
    b = numpy.array([[weight(G[i]) % 2] for i in range(G.shape[0])])
    G = numpy.append(G, b, axis=1)
    return G, H


def test_errors(G: Array, H: Array, syndromes: TableOfSyndromes, errors_count: int) -> None:
    k, n = G.shape
    if errors_count > n:
        print(f'Количесво ошибок превышает длину кодовых слов ({errors_count} > {n})')
        return
    print('Тест количества ошибок:', errors_count)
    print('-------------------------')
    word = numpy.random.randint(0, 2, k)
    print('Слово:', word)
    message = dot(word, G)
    print('Сообщение:', message)
    error = numpy.zeros(n, dtype=int)
    possible_error_positions = [i for i in range(n)]
    for _ in range(errors_count):
        error[possible_error_positions.pop(random.randint(0, len(possible_error_positions) - 1))] = 1
    print('Вектор ошибки:', error)
    message_with_error = add(message, error)
    print('Сообщение с ошибками:', message_with_error)
    syndrome = dot(message_with_error, H)
    print('Синдром:', syndrome)
    calculated_message = message
    if weight(syndrome) == 0:
        print('Ошибок не обнаружено')
    else:
        syndrome_as_tuple = tuple(syndrome)
        if syndrome_as_tuple not in syndromes:
            print('Синдром не найден в таблице синдромов -> Невозможно исправить ошибку -> Повторный запрос сообщения')
            print(f'Код позволяет обнаружить количество ошибок: {errors_count}, но не позволяет их исправить')
            return
        print('Синдром найден в таблице синдромов -> Попытка исправить ошибку')
        calculated_error = syndromes[syndrome_as_tuple]
        print('Вычисленный вектор ошибки:', calculated_error,
              f'(количество найденных ошибок: {weight(calculated_error)})')
        calculated_message = add(message_with_error, calculated_error)
        print('Исправленное сообщение:', message)
    calculated_word = calculated_message[:k]
    print('Декодированное слово:', calculated_word)
    if weight(add(calculated_word, word)) == 0:
        print('Декодированное слово совпадает с исходным, код позволяет исправить количество ошибок:', errors_count)
    else:
        print('Декодированное слово не совпадает с исходным, код не позволяет найти или исправить количество ошибок:',
              errors_count)


def main() -> None:
    codes: list[Code] = []
    for r in range(2, 5):
        codes.append(create_Hamming_code(r))

    for k, (G, H) in enumerate(codes):
        print('----------------------------------')
        print(f'Исследование кода Хемминга с r = {k + 2}')
        print('----------------------------------')
        print()
        print('G:')
        print(G)
        print('H:')
        print(H)
        print()
        syndromes = create_table_of_syndromes(H, 1)
        for i in range(1, 4):
            test_errors(G, H, syndromes, i)
            print()

    extended_codes: list[Code] = []
    for code in codes:
        extended_codes.append(extend_Hamming_code(*code))

    for k, (G, H) in enumerate(extended_codes):
        print('-----------------------------------------------')
        print(f'Исследование расширенного кода Хемминга с r = {k + 2}')
        print('-----------------------------------------------')
        print()
        print('G*:')
        print(G)
        print('H*:')
        print(H)
        print()
        syndromes = create_table_of_syndromes(H, 1)
        for i in range(1, 5):
            test_errors(G, H, syndromes, i)
            print()


if __name__ == '__main__':
    main()
