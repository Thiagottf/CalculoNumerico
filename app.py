from flask import Flask, render_template, request
import numpy as np
from sympy import symbols, simplify

app = Flask(__name__)

# Suas funções
def jacobi(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum = 0
            for j in range(n):
                if i != j:
                    sum += A[i, j] * x[j]
            x_new[i] = (b[i] - sum) / A[i, i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    
    raise Exception(f'Não convergiu após {max_iterations} iterações')

def gauss_seidel(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    
    raise Exception(f'Não convergiu após {max_iterations} iterações')

def eliminacao_gauss(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] = Ab[j] - factor * Ab[i]
    
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

def is_diagonally_dominant(A):
    for i in range(len(A)):
        row_sum = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if abs(A[i, i]) < row_sum:
            return False
    return True

def tornar_diagonalmente_dominante(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            if abs(A[j, i]) >= sum(abs(A[j, k]) for k in range(n) if k != i):
                A[[i, j]] = A[[j, i]]
                b[[i, j]] = b[[j, i]]
                break

    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            return A, b, False
    return A, b, True

def lagrange_interpolation(x_values, y_values, x_to_evaluate):
    x = symbols('x')
    n = len(x_values)
    polynomial = 0
    
    for j in range(n):
        term = y_values[j]
        for m in range(n):
            if m != j:
                term *= (x - x_values[m]) / (x_values[j] - x_values[m])
        polynomial += term

    polynomial = simplify(polynomial)
    result = polynomial.subs(x, x_to_evaluate)
    
    return result, polynomial

def resolver_sistema_linear(opcao, A, b, x0, tol, max_iterations):
    if opcao == 1:
        return jacobi(A, b, x0, tol, max_iterations)
    elif opcao == 2:
        return gauss_seidel(A, b, x0, tol, max_iterations)
    else:
        raise ValueError("Opção inválida. Escolha 1 para 'jacobi' ou 2 para 'gauss-seidel'.")

# Rotas do Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sistema_linear', methods=['GET', 'POST'])
def sistema_linear():
    if request.method == 'POST':
        n = int(request.form['n'])
        A = np.array([[float(request.form[f'a{i}{j}']) for j in range(n)] for i in range(n)])
        b = np.array([float(request.form[f'b{i}']) for i in range(n)])
        x0 = np.array([float(request.form[f'x0{i}']) for i in range(n)])
        tol = float(request.form['tol'])
        max_iterations = int(request.form['max_iterations'])
        metodo = int(request.form['metodo'])

        A, b, sucesso = tornar_diagonalmente_dominante(A, b)

        if sucesso:
            try:
                x, iterations = resolver_sistema_linear(metodo, A, b, x0, tol, max_iterations)
                metodo_nome = 'Jacobi' if metodo == 1 else 'Gauss-Seidel'
                return render_template('sistema_linear.html', resultado=f'Solução: {x}, Iterações: {iterations}', metodo=metodo_nome)
            except Exception as e:
                return render_template('sistema_linear.html', erro=str(e))
        else:
            return render_template('sistema_linear.html', erro="A matriz não é diagonalmente dominante e o método pode não convergir.")
    return render_template('sistema_linear.html')

@app.route('/interpolacao', methods=['GET', 'POST'])
def interpolacao():
    if request.method == 'POST':
        n = int(request.form['n'])
        x_values = [float(request.form[f'x{i}']) for i in range(n)]
        y_values = [float(request.form[f'y{i}']) for i in range(n)]
        x_to_evaluate = float(request.form['x_to_evaluate'])

        result, polynomial = lagrange_interpolation(x_values, y_values, x_to_evaluate)
        return render_template('interpolacao.html', resultado=f'O valor interpolado de f({x_to_evaluate}) é {result}', polinomio=f'Polinômio: {polynomial}')
    return render_template('interpolacao.html')

if __name__ == "__main__":
    app.run(debug=True)
