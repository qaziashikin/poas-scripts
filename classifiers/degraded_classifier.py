'''
    Classification of degraded images. Conditional classifier - run prior to other detectors.
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class DegradedClassifier:
    def __init__(self):
        """
        Initialize the class DegradedClassifier.
        """

    def plot_curve_for_pixel_means(column_averages):
        x = np.array(range(16))
        y = np.array(column_averages)

        X = x.reshape(-1, 1)

        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        X_smooth = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        X_smooth_poly = poly_features.transform(X_smooth)
        y_smooth = model.predict(X_smooth_poly)

        r2 = model.score(X_poly, y)

        a = model.coef_[2]
        b = model.coef_[1]
        c = model.intercept_

        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, color='blue', label='Original data points')
        plt.plot(X_smooth, y_smooth, color='red', label='Quadratic fit')
        plt.title('Average Pixel Values with Quadratic Fit')
        plt.xlabel('Column Section (width 40 pixels)')
        plt.ylabel('Average Pixel Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        equation = f'y = {a:.2f}x² + {b:.2f}x + {c:.2f}'
        plt.text(0.02, 0.95, f'Equation: {equation}\nR² = {r2:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))

        plt.show()

        print(f"Quadratic equation: {equation}")
        print(f"R-squared value: {r2:.4f}")

        return a, b, c
    

    def calculate_column_pixel_means(self, image, column_width=40):
        num_columns = image.shape[1] // column_width
        
        column_averages = []
        
        for i in range(num_columns):
            start_col = i * column_width
            end_col = start_col + column_width
            column_section = image[:, start_col:end_col]
            average = np.mean(column_section)
            column_averages.append(average)
        
        return column_averages


    def get_coefficients_for_linear_regression(self, column_averages):
        x = np.array(range(16))
        y = np.array(column_averages)

        X = x.reshape(-1, 1)

        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        a = model.coef_[2]
        b = model.coef_[1]
        c = model.intercept_

        return a, b, c


    def is_degraded(self, img):
        blurred_img = cv.GaussianBlur(img, (35, 35), 15)
        a, _, _ = self.get_coefficients_for_linear_regression(self.calculate_column_pixel_means(blurred_img))
        return a > 0.4
