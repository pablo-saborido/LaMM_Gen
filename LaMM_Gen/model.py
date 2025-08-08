import numpy as np
import matplotlib.pyplot as plt
import inspect
from sklearn.decomposition import PCA
from scipy.linalg import cholesky

class LMM:

    @staticmethod
    def unit_ball(n, d):
      # pick random directions on the (d–1)-sphere
      X = np.random.randn(n, d)
      X /= np.linalg.norm(X, axis=1, keepdims=True)
      # pick radii so that points fill the volume uniformly
      u = np.random.rand(n)         # uniform in [0,1]
      r = u**(1.0 / d)              # scaling to get uniform-in-volume
      return X * r[:, None]

    @staticmethod
    def cylinder(n, height=1.0):
      θ = 2*np.pi*np.random.rand(n)
      z = np.random.rand(n)*height
      return np.vstack((np.cos(θ), np.sin(θ), z)).T

    @staticmethod
    def swiss_roll(n):
      t = (3*np.pi/2)*(1 + 2*np.random.rand(n))
      x = t*np.cos(t)
      y = 21*np.random.rand(n)
      z = t*np.sin(t)
      data = np.vstack((x, y, z)).T
      return data

    @staticmethod
    def s_curve(n):
      t = 3*np.pi*(np.random.rand(n) - 0.5)
      x = np.sin(t)
      y = 2*np.random.rand(n)
      z = np.sign(t)*(1 - np.cos(t))
      data = np.vstack((x, y, z)).T
      return data


    # Dictionary of built-in latent-space generators, n is always the number of samples
    Latent_dictionary = {
        "unit interval": lambda n: np.linspace(0, 1, n).reshape(-1, 1),
        "unit circle": lambda n: np.vstack((
            np.cos(2 * np.pi * np.random.rand(n)),
            np.sin(2 * np.pi * np.random.rand(n))
        )).T,
        "figure eight" : lambda n: (
        # sample n random parameters t in [0, 2π)
            (lambda t: np.vstack((np.cos(t),
                                  np.sin(t) * np.cos(t))).T)
            (2 * np.pi * np.random.rand(n))
        ),
        "unit sphere": lambda n, d: (
            (lambda X: X / np.linalg.norm(X, axis=1, keepdims=True))
            (np.random.randn(n, d))
        ),
        "ball": lambda n, d=2: LMM.unit_ball(n, d),
        "cylinder": lambda n, height=1.0: LMM.cylinder(n, height),
        "swiss roll": lambda n: LMM.swiss_roll(n),
        "s curve": lambda n: LMM.s_curve(n),
        # Torus embedded in R^3, with major radius R and minor radius r
        "torus": lambda n, R=1.0, r=0.2: (
            (lambda u, v: np.vstack((
                (R + r * np.cos(v)) * np.cos(u),
                (R + r * np.cos(v)) * np.sin(u),
                r * np.sin(v)
            )).T)(
                2*np.pi*np.random.rand(n),
                2*np.pi*np.random.rand(n)
            )
        ),
    }

    # Dictionary of built-in kernels
    Kernel_dictionary = {
        # polynomial: (⟨x,y⟩ + 1)^2
        "polynomial": lambda x, y: (np.inner(x, y) + 1.0) ** 2,
        # RBF, radial basis function: exp(−‖x−y‖²/2)
        "RBF": lambda x, y: np.exp(-0.5 * np.linalg.norm(x - y) ** 2),
        # random: random mxm positive definite matrix, only possible for latent spaces of dimension 0, with m possible elements.
        "random": None, # The actual kernel will be computen in the __init__
    }

    def __init__(self, Z, covariance_kernel, sigma=0.0, eps=1e-8, n_latent=None, latent_params=None):
        """
         - Z (str or (n,d) array): latent space samples. If string, one of LMM.Latent_dictionary keys, else: custom latent samples whith
                                n samples, each of which is a d-dimensional point.
         - covariance_kernel (str or callable): if string, one of LMM.Kernel_dictionary keys, or a custom callable f(x,y)->float.
         - sigma (float): standard deviation of the noise.
         - eps (float): small value to add to the kernel matrix before Cholesky decomposition for numerical stability.
         - n_latent (int): number of samples in the latent space, required if Z is a built-in latent-space name.
         - latent_params (tuple or None): parameters required by the built-in latent sample generator.
        """

        self.sigma = sigma

        # ---Store latent samples---
        self.Z = generate_latent_space(Z, n_latent=n_latent, latent_params=latent_params)
        self.n, self.d = self.Z.shape

        # ---Store the kernel---
        if isinstance(covariance_kernel, str):
            # If it's not a built-in kernel, raise error
            if covariance_kernel not in LMM.Kernel_dictionary:
                raise ValueError(
                    f"Unknown kernel '{covariance_kernel}'. "
                    f"Available: {list(LMM.Kernel_dictionary)}"
                )
            # If it's not a built-in kernel, raise error

            if covariance_kernel == "random":
                # Build a random positive definite matrix
                A = np.random.randn(self.n, self.n)
                K = A @ A.T # This is always positive definite
                # Add terms in the diagonal to prevent numerical errors
                K[np.diag_indices(self.n)] += eps * self.n
                self.L = cholesky(K, lower=True)
                return

            # If it's in the predefined dictionary and is not random, just pick that function
            self.kernel = LMM.Kernel_dictionary[covariance_kernel]

        elif callable(covariance_kernel):
            # If a function is provided just store it
            self.kernel = covariance_kernel

        else:
            raise TypeError("covariance_kernel must be a string or a callable.")

        # ---Build kernel matrix and factorise---
        # If the kernel is not random, we still need to compute the kernel matrix K and its Cholesky decomposition
        K = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i, j] = self.kernel(self.Z[i], self.Z[j])
        # Add terms in the diagonal to ensure positive definiteness and prevent numerical errors
        K[np.diag_indices(self.n)] += eps

        # Store the Cholesky factor L: K = L @ L.T
        self.L = cholesky(K, lower=True)
    
    @staticmethod
    def generate_latent_space(Z, n_latent=None, latent_params=None):
        """
        Resolve `Z` into an (n,d) numpy array.
        Arguments:
            - Z (str or (n,d) array): latent space samples. If string, one of LMM.Latent_dictionary keys, else: custom latent 
                                      samples whith n samples, each of which is a d-dimensional point.
            - n_latent (int): number of samples in the latent space, required if Z is a built-in latent-space name.
            - latent_params (tuple or None): parameters required by the built-in latent sample generator.
        Returns:
            - ndarray of shape (n, d).
        """
        # If user tries to use a built-in latent generator
        if isinstance(Z, str):
            # Check if the generator exists
            try:
                generator = LMM.Latent_dictionary[Z]
            except KeyError:
                raise ValueError(f"Unknown latent space '{Z}'. "
                                 f"Available: {list(LMM.Latent_dictionary)}")
            # The number of samples has to be provided
            if n_latent is None:
                raise ValueError("You must provide 'n_latent' when using a built-in latent space.")

            # Now check that all the required arguemnts for the generator are provided

            # First, inspect generator signature
            sig = inspect.signature(generator)
            # Parameters excluding the first 'n', already given by 'n_latent'
            params = list(sig.parameters.values())[1:]
            # Identify required parameter names
            required = tuple(p.name for p in params if p.default is inspect.Parameter.empty)

            # Normalise a single int into a singleton tuple
            if isinstance(latent_params, int):
                latent_params = (latent_params,)
            # Normalise list into a tuple
            elif isinstance(latent_params, list):
                latent_params = tuple(latent_params)
            # If latent_params is None we will have an empty tuple
            elif latent_params is None:
                latent_params = ()
            # Reject anything else
            elif not isinstance(latent_params, tuple):
                raise TypeError(f"latent_params must be an int, list or tuple; got {type(latent_params).__name__}")

            # Check that the user provided enough arguments
            if len(latent_params) < len(required):
                raise ValueError(
                    f"Generator '{Z}' requires parameters {required}, got {latent_params}"
                )

            # Call generator with n_latent and provided args
            return np.asarray(generator(n_latent, *latent_params))

        # If user tries to provide a custom set of samples
        else:
            try:
                # If Z is array-like
                Z_arr = np.asarray(Z)
            except (TypeError, ValueError) as e:
                raise TypeError("Z must be a string or array-like.") from e
            
            # Check that the shape of the array is the one we expect
            if Z_arr.ndim != 2:
                raise ValueError("Custom latent samples must be 2D (n, d).")
            
            return Z_arr         


    def generate(self, n_features):
        """
        Generate the dataset.
        Arguments:
            - n_features (int): Number of features (columns) of the dataset we want to generate.
        Returns:
            - Y ((n,n_features) array): Y[:, j] = X_j(Z) + sigma * E_j, where each X_j ~ N(0,K) (or random kernel), E_j ~ N(0,I).
        """

        # Generate the dataset from a matrix of standard normal random variables
        W = np.random.randn(self.n, n_features)
        X = self.L @ W

        if self.sigma > 0:
            E = np.random.randn(self.n, n_features)
            self.Y = X + self.sigma * E
        else:
            self.Y = X
        return self.Y

    def plot_dataset(self):
        """
        Scatter-plot the last generated dataset if available and dimensionality less than 4.
        """
        if self.Y is None:
            raise RuntimeError("No dataset generated yet, call generate() first.")
        Y = self.Y
        n, p = Y.shape
        if p > 3:
            raise ValueError("Cannot plot dataset with more than 3 features.")

        if p == 1:
            plt.figure()
            plt.plot(Y[:, 0], marker='o', linestyle='')
            plt.xlabel('Sample index')
            plt.ylabel('Feature 0')
            plt.title('Generated data (1D)')
            plt.show()

        elif p == 2:
            plt.figure()
            plt.scatter(Y[:, 0], Y[:, 1], alpha=0.7)
            plt.xlabel('Feature 0')
            plt.ylabel('Feature 1')
            plt.title('Generated data (2D)')
            plt.axis('equal')
            plt.show()

        else:  # p == 3
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], alpha=0.7)
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Feature 1')
            ax.set_zlabel('Feature 2')
            ax.set_title('Generated data (3D)')
            plt.show()

    def plot_latent(self):
        """
        Scatter-plot the latent-space points Z if dimensionality less than 4.
        """
        n, d = self.Z.shape
        if d > 3:
            raise ValueError("Cannot plot latent space with dimension > 3.")

        if d == 1:
            plt.figure()
            plt.plot(self.Z[:, 0], np.zeros(n), marker='o', linestyle='')
            plt.xlabel('Dimension 0')
            plt.title('Latent space (1D)')
            plt.yticks([])
            plt.show()

        elif d == 2:
            plt.figure()
            plt.scatter(self.Z[:, 0], self.Z[:, 1], alpha=0.7)
            plt.xlabel('Dimension 0')
            plt.ylabel('Dimension 1')
            plt.title('Latent space (2D)')
            plt.axis('equal')
            plt.show()

        else:  # d == 3
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.Z[:, 0], self.Z[:, 1], self.Z[:, 2], alpha=0.7)
            ax.set_xlabel('Dimension 0')
            ax.set_ylabel('Dimension 1')
            ax.set_zlabel('Dimension 2')
            ax.set_title('Latent space (3D)')
            # ax.set_aspect('equal')

            # Calculate the actual data ranges for each axis
            x_range = self.Z[:, 0].max() - self.Z[:, 0].min()
            y_range = self.Z[:, 1].max() - self.Z[:, 1].min()
            z_range = self.Z[:, 2].max() - self.Z[:, 2].min()

            # Find the maximum range to use as reference
            max_range = max(x_range, y_range, z_range)

            # Calculate the center point for each axis
            x_center = (self.Z[:, 0].max() + self.Z[:, 0].min()) / 2
            y_center = (self.Z[:, 1].max() + self.Z[:, 1].min()) / 2
            z_center = (self.Z[:, 2].max() + self.Z[:, 2].min()) / 2

            # Add a small margin (e.g., 10%) for better visualization
            margin = max_range * 0.1
            half_range = (max_range + margin) / 2

            # Set symmetric limits around the center, all with the same range
            ax.set_xlim([x_center - half_range, x_center + half_range])
            ax.set_ylim([y_center - half_range, y_center + half_range])
            ax.set_zlim([z_center - half_range, z_center + half_range])

            plt.show()
    def plot_pca_dataset(self):
        """
        Apply PCA (3 components) to the last generated dataset and plot the result.
        Only valid when the dataset has >3 features.
        """
        if self._Y is None:
            raise RuntimeError("No dataset generated yet; call generate() first.")
        Y = self._Y
        n, p = Y.shape

        # check: only use PCA‐plot when p > 3
        if p <= 3:
            raise ValueError(f"Dataset has only {p} features; use plot_dataset() instead of plot_pca_dataset().")

        # Perform PCA to 3 components
        pca = PCA(n_components=3)
        Y_pca = pca.fit_transform(Y)

        # Now plot the 3D PCA scatter
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Y_pca[:, 0], Y_pca[:, 1], Y_pca[:, 2], alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA (3 components) of dataset")
        plt.show()