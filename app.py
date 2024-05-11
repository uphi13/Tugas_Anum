from flask import Flask, request, render_template
import sympy as sp
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        method = request.form.get('method')
        if method == 'newton-raphson':
            return newton_raphson()
        elif method == 'secant':
            return secant()
    return render_template('index.html')

@app.route('/newton_raphson', methods=['GET', 'POST'])
def newton_raphson():
    if request.method == 'POST':
        f_x = request.form.get('f_x')
        if f_x is None or f_x.strip() == '':
            return render_template('newton-raphson.html')
        f_x = f_x.replace('^', '**')
        x_n = float(request.form.get('x_n'))
        epsilon = float(request.form.get('epsilon'))

        x = sp.symbols('x')
        f = sp.sympify(f_x)
        f_prime = sp.diff(f, x)

        Xn = [x_n]
        f_Xn = [f.subs(x, x_n)]
        f_prime_Xn = [f_prime.subs(x, x_n)]
        errors = [np.nan]

        while True:
            x_n1 = x_n - f.subs(x, x_n) / f_prime.subs(x, x_n)
            error = abs(x_n1 - x_n)
            Xn.append(x_n1)
            f_Xn.append(f.subs(x, x_n1))
            f_prime_Xn.append(f_prime.subs(x, x_n1))
            errors.append(error)
            if error < epsilon:
                break
            x_n = x_n1

        df = pd.DataFrame(list(zip(Xn, f_Xn, f_prime_Xn, errors)), columns=['Xn', 'f(Xn)', 'f\'(Xn)', 'Error'])
        return render_template('table.html', tables=[df.to_html(classes='data', header="true", index="false")], titles=df.columns.values, result=x_n1, method='Newton-Raphson')

    return render_template('newton-raphson.html')

@app.route('/secant', methods=['GET', 'POST'])
def secant():
    if request.method == 'POST':
        f_x = request.form.get('f_x')
        if f_x is None or f_x.strip() == '':
            return render_template('secant.html')
        f_x = f_x.replace('^', '**')
        x_n_minus_1 = float(request.form.get('x_n_minus_1'))
        x_n = float(request.form.get('x_n'))
        epsilon = float(request.form.get('epsilon'))

        x = sp.symbols('x')
        f = sp.sympify(f_x)

        X0 = [x_n_minus_1]
        X1 = [x_n]
        f_X0 = [f.subs(x, x_n_minus_1)]
        f_X1 = [f.subs(x, x_n)]
        X2 = [None]
        f_X2 = [None]

        while True:
            x_n_plus_1 = x_n - (f.subs(x, x_n) * (x_n - x_n_minus_1)) / (f.subs(x, x_n) - f.subs(x, x_n_minus_1))
            X2[-1] = x_n_plus_1
            f_X2[-1] = f.subs(x, x_n_plus_1)
            if abs(f_X2[-1]) < epsilon:
                break
            x_n_minus_1 = x_n
            x_n = x_n_plus_1
            X0.append(x_n_minus_1)
            X1.append(x_n)
            f_X0.append(f.subs(x, x_n_minus_1))
            f_X1.append(f.subs(x, x_n))
            X2.append(None)
            f_X2.append(None)

        df = pd.DataFrame(list(zip(X0, X1, f_X0, f_X1, X2, f_X2)), columns=['X0', 'X1', 'f(X0)', 'f(X1)', 'X2', 'f(X2)'])
        return render_template('table.html', tables=[df.to_html(classes='data', header="true", index="false")], titles=df.columns.values, result=x_n_plus_1, method='Secant')

    return render_template('secant.html')

if __name__ == '__main__':
    app.run(debug=True)