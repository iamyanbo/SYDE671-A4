# Yanbo Cheng
# 21287485

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from skimage.io import imread
from skimage.transform import warp

from sfm_utils import Frame, bundle_adjustment, merge_all_frames


def get_data_from_txt_file(filename):
    with open(filename) as file:
        values = np.loadtxt(file, skiprows=1, ndmin=2)
    if values.shape[1] == 2:
        # Read (y, x) as (x, y, 1)
        return np.column_stack((values[:, 1], values[:, 0], np.ones(len(values))))
    return values


def lls_eight_point_alg(points1, points2):
    points1, points2 = np.asarray(points1, dtype=float), np.asarray(points2, dtype=float)
    points1 = points1 / points1[:, 2:3]
    points2 = points2 / points2[:, 2:3]

    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]
    # Build one equation per match
    A = np.column_stack((x2*x1, x2*y1, x2,
                         y2*x1, y2*y1, y2,
                         x1, y1, np.ones(len(points1))))
    # Keep all nine right singular vectors even with eight matches
    _, _, Vh = np.linalg.svd(A, full_matrices=True)
    F = Vh[-1].reshape(3, 3)

    # Enforce rank two
    U, singular_values, Vh = np.linalg.svd(F)
    singular_values[-1] = 0.0
    F = U @ np.diag(singular_values) @ Vh
    # Fix the scale and sign
    entries = F.ravel()
    largest = entries[np.argmax(np.abs(entries))]
    F = F / np.linalg.norm(F)
    return -F if largest < 0 else F


def normalized_eight_point_alg(points1, points2):
    points1, points2 = np.asarray(points1, dtype=float), np.asarray(points2, dtype=float)
    points1 = points1 / points1[:, 2:3]
    points2 = points2 / points2[:, 2:3]

    def normalize(points):
        center = points[:, :2].mean(axis=0)
        centered = points[:, :2] - center
        mean_distance = np.linalg.norm(centered, axis=1).mean()
        # Set the mean radius to sqrt(2)
        scale = np.sqrt(2.0) / mean_distance
        T = np.array([[scale, 0.0, -scale*center[0]],
                      [0.0, scale, -scale*center[1]],
                      [0.0, 0.0, 1.0]])
        return (T @ points.T).T, T

    normalized1, T1 = normalize(points1)
    normalized2, T2 = normalize(points2)
    F_normalized = lls_eight_point_alg(normalized1, normalized2)
    # Convert F back to pixel coordinates
    F = T2.T @ F_normalized @ T1
    entries = F.ravel()
    largest = entries[np.argmax(np.abs(entries))]
    F = F / np.linalg.norm(F)
    return -F if largest < 0 else F


def compute_distance_to_epipolar_lines(points1, points2, F):
    points1, points2 = np.asarray(points1, dtype=float), np.asarray(points2, dtype=float)
    points1 = points1 / points1[:, 2:3]
    points2 = points2 / points2[:, 2:3]

    # Epipolar lines in image 1
    lines = points2 @ F
    a, b, c = lines.T
    x, y = points1[:, :2].T
    distances = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    return float(distances.mean())


def compute_epipole(F):
    # Solve F.T @ e2 = 0
    _, _, Vh = np.linalg.svd(np.asarray(F, dtype=float).T)
    return Vh[-1]


def compute_H(e, im):
    e = np.asarray(e, dtype=float).reshape(3)
    height, width = im.shape[:2]
    T = np.array([[1.0, 0.0, -width/2.0],
                  [0.0, 1.0, -height/2.0],
                  [0.0, 0.0, 1.0]])
    translated = T @ e
    ex, ey = translated[:2]
    radius = np.sqrt(ex**2 + ey**2)
    # Rotate the epipole onto the x-axis
    alpha = 1.0 if ex >= 0 else -1.0
    cos_theta = alpha * ex / radius
    sin_theta = alpha * ey / radius
    R = np.array([[cos_theta, sin_theta, 0.0],
                  [-sin_theta, cos_theta, 0.0],
                  [0.0, 0.0, 1.0]])
    rotated = R @ translated
    G = np.eye(3)
    # Send the epipole to infinity
    G[2, 0] = -rotated[2] / rotated[0]
    return np.linalg.inv(T) @ G @ R @ T


def compute_matching_homographies(e2, F, im2, points1, points2):
    e2 = np.asarray(e2, dtype=float).reshape(3)
    points1, points2 = np.asarray(points1, dtype=float), np.asarray(points2, dtype=float)
    points1 = points1 / points1[:, 2:3]
    points2 = points2 / points2[:, 2:3]
    H2 = compute_H(e2, im2)
    ex, ey, ew = e2
    cross_e = np.array([[0.0, -ew, ey],
                        [ew, 0.0, -ex],
                        [-ey, ex, 0.0]])
    v = np.array([[1, 1, 1]])
    M = cross_e @ F + e2.reshape(3, 1) @ v

    points1_hat = (H2 @ M @ points1.T).T
    points2_hat = (H2 @ points2.T).T
    W = points1_hat / points1_hat[:, 2:3]
    b = points2_hat[:, 0] / points2_hat[:, 2]
    a = np.linalg.lstsq(W, b, rcond=None)[0]
    HA = np.eye(3)
    # Adjust x without changing the rows
    HA[0, :] = a
    H1 = HA @ H2 @ M
    return H1, H2


def factorization_method(points_im1, points_im2):
    points_im1 = np.asarray(points_im1, dtype=float)
    points_im2 = np.asarray(points_im2, dtype=float)
    points_im1 = points_im1 / points_im1[:, 2:3]
    points_im2 = points_im2 / points_im2[:, 2:3]

    # One column per match
    W = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T))
    # Remove image translation
    W = W - W.mean(axis=1, keepdims=True)
    U, singular_values, Vh = np.linalg.svd(W, full_matrices=False)

    # Keep three singular values
    sqrt_sigma = np.diag(np.sqrt(singular_values[:3]))
    motion = U[:, :3] @ sqrt_sigma       # 4 x 3
    structure = sqrt_sigma @ Vh[:3, :]  # 3 x N

    # Fit L = Q @ Q.T using two lengths and one dot product per camera
    camera_rows = motion.reshape(2, 2, 3)
    equations = []
    for r, s in camera_rows:
        for left, right in ((r, r), (r, s), (s, s)):
            x1, y1, z1 = left
            x2, y2, z2 = right
            equations.append([x1*x2, x1*y2 + y1*x2, x1*z2 + z1*x2,
                              y1*y2, y1*z2 + z1*y2, z1*z2])
    target = [1, 0, 1, 1, 0, 1]
    xx, xy, xz, yy, yz, zz = np.linalg.lstsq(equations, target, rcond=None)[0]
    L = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    # Factor L and correct both SVD matrices
    Q = np.linalg.cholesky(L)
    return np.linalg.solve(Q, structure), motion @ Q


def similarity_alignment(reconstruction, truth):
    # Align to ground truth for the plot
    X = reconstruction - reconstruction.mean(axis=0)
    Y = truth - truth.mean(axis=0)
    U, values, Vh = np.linalg.svd(X.T @ Y)
    Q = U @ Vh
    scale = values.sum() / np.sum(X*X)
    return scale * X @ Q, Y


def estimate_initial_RT(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    translation = U[:, 2]

    poses = []
    for rotation in (R1, R2):
        # Force determinant +1
        rotation = np.linalg.det(rotation) * rotation
        for direction in (translation, -translation):
            poses.append(np.column_stack((rotation, direction)))
    return np.stack(poses)


def linear_estimate_3d_point(image_points, camera_matrices):
    rows = np.empty((2 * len(image_points), 4))
    for i, (u, v) in enumerate(image_points[:, :2]):
        m1, m2, m3 = camera_matrices[i]
        rows[2*i] = u * m3 - m1
        rows[2*i + 1] = v * m3 - m2
    _, _, Vt = np.linalg.svd(rows, full_matrices=False)
    point = Vt[-1]
    return point[:3] / point[3]


def reprojection_error(point_3d, image_points, camera_matrices):
    projected = camera_matrices @ np.append(point_3d, 1.0)
    pixels = projected[:, :2] / projected[:, 2:3]
    return (pixels - image_points).ravel()


def jacobian(point_3d, camera_matrices):
    projected = camera_matrices @ np.append(point_3d, 1.0)
    J = np.empty((2 * len(camera_matrices), 3), dtype=projected.dtype)
    for i, camera in enumerate(camera_matrices):
        u, v, w = projected[i]
        m1, m2, m3 = camera[:, :3]

        # Differentiate the projected x and y coordinates
        w_squared = w*w
        J[2*i] = (w*m1 - u*m3) / w_squared
        J[2*i + 1] = (w*m2 - v*m3) / w_squared
    return J


def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        error = reprojection_error(point, image_points, camera_matrices)
        derivative = jacobian(point, camera_matrices)
        step = np.linalg.lstsq(derivative, -error, rcond=None)[0]
        point += step
    return point


def estimate_RT_from_E(E, image_points, K):
    first_camera = K @ np.eye(3, 4)
    best_pose = None
    best_count = -1
    for pose in estimate_initial_RT(E):
        cameras = np.stack((first_camera, K @ pose))
        rotation, translation = pose[:, :3], pose[:, 3]
        count = 0
        for match in image_points:
            point = nonlinear_estimate_3d_point(match, cameras)
            second_point = rotation @ point + translation
            # Positive depth in both cameras
            if point[2] > 0 and second_point[2] > 0:
                count += 1
        if count > best_count:
            best_pose = pose
            best_count = count
    return best_pose


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    out_dir = project_dir / "results"
    out_dir.mkdir(exist_ok=True)

    for name in ("set1", "set2"):
        data = data_dir / name
        points1 = get_data_from_txt_file(data / "pt_2D_1.txt")
        points2 = get_data_from_txt_file(data / "pt_2D_2.txt")

        # 1.a
        F = lls_eight_point_alg(points1, points2)

        # 1.b
        print(name, "F =\n", F)

        # 1.c
        F = normalized_eight_point_alg(points1, points2)

        # 1.d
        print(name, "normalized F =\n", F)

    # 2.a
    data = data_dir / "set1"
    points1 = get_data_from_txt_file(data / "pt_2D_1.txt")
    points2 = get_data_from_txt_file(data / "pt_2D_2.txt")
    F = normalized_eight_point_alg(points1, points2)
    e2 = compute_epipole(F)

    # 2.b, 2.c
    im2 = imread(data / "image2.jpg")
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)

    # 2.d
    im1 = imread(data / "image1.jpg")
    images = (im1, im2)
    bounds = []
    for H, im in zip((H1, H2), images):
        height, width = im.shape[:2]
        corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                            [width - 1, height - 1, 1], [0, height - 1, 1]])
        corners = (H @ corners.T).T
        bounds.append(corners[:, :2] / corners[:, 2:3])

    # Use the same vertical origin in both rectified images
    y_min = int(np.floor(min(box[:, 1].min() for box in bounds)))
    y_max = int(np.ceil(max(box[:, 1].max() for box in bounds)))
    height = y_max - y_min + 1
    rectified = []
    rectified_points = []
    for H, im, box, points in zip((H1, H2), images, bounds, (points1, points2)):
        x_min = int(np.floor(box[:, 0].min()))
        x_max = int(np.ceil(box[:, 0].max()))
        shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H = shift @ H
        rectified.append(warp(im, inverse_map=np.linalg.inv(H),
                              output_shape=(height, x_max - x_min + 1), preserve_range=True))
        points = (H @ points.T).T
        rectified_points.append(points / points[:, 2:3])

    widths = [im.shape[1] for im in rectified]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": widths})
    for i, (ax, im, points) in enumerate(zip(axes, rectified, rectified_points), 1):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        for y in range(80, height, 80):
            ax.axhline(y, color="#f1bb55", lw=0.65, alpha=0.85)
        ax.scatter(points[:, 0], points[:, 1], color="#16bcff", marker="+", s=22, linewidths=0.9)
        ax.set_xlim(0, im.shape[1] - 1)
        ax.set_ylim(height - 1, 0)
        ax.set_title(f"Image {i}", fontsize=12)
        ax.axis("off")
    fig.suptitle("Rectified images", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, "Blue crosses: matched points   |   Gold: horizontal row guides",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.07, top=0.91, wspace=0.035)
    fig.savefig(out_dir / "rectified_pair.png", dpi=160, facecolor="white")
    plt.close(fig)

    for name in ("set1", "set1_subset"):
        data = data_dir / name
        points1 = get_data_from_txt_file(data / "pt_2D_1.txt")
        points2 = get_data_from_txt_file(data / "pt_2D_2.txt")

        W = np.vstack((points1[:, :2].T, points2[:, :2].T))
        W -= W.mean(axis=1, keepdims=True)
        _, singular_values, Vh = np.linalg.svd(W, full_matrices=False)

        if name == "set1":
            # 3.a
            structure, _ = factorization_method(points1, points2)
            reconstruction_label = "Reconstruction"
        else:
            # 3.e
            # The subset's least-squares L has a negative eigenvalue.
            structure = np.diag(np.sqrt(singular_values[:3])) @ Vh[:3]
            reconstruction_label = "Initial SVD shape"

        # 3.c, 3.e
        truth = get_data_from_txt_file(data / "pt_3D.txt")
        aligned, centered_truth = similarity_alignment(structure.T, truth)
        error = np.linalg.norm(aligned - centered_truth) / np.linalg.norm(centered_truth)
        print(f"{name} alignment error = {error:.2%}")

        # Equal axis limits for both views
        combined = np.vstack((aligned, centered_truth))
        center = (combined.min(axis=0) + combined.max(axis=0)) / 2
        radius = np.ptp(combined, axis=0).max() * 0.55
        fig = plt.figure(figsize=(12, 6))
        for i, (azimuth, title) in enumerate([(-90, "Front view"), (0, "Side view")], 1):
            ax = fig.add_subplot(1, 2, i, projection="3d")
            ax.scatter(*centered_truth.T, s=40, color="#1670bf", alpha=0.8, label="Ground truth")
            ax.scatter(*aligned.T, s=35, marker="x", color="#da6a13", alpha=0.9, label=reconstruction_label)
            ax.set(xlim=(center[0] - radius, center[0] + radius),
                   ylim=(center[1] - radius, center[1] + radius),
                   zlim=(center[2] - radius, center[2] + radius))
            ax.set_box_aspect((1, 1, 1))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(4))
            ax.view_init(elev=12, azim=azimuth)
            ax.set(xlabel="X", ylabel="Y", zlabel="Z")
            ax.set_title(title, pad=8)
            if i == 1:
                ax.legend(loc="upper left", fontsize=9)
        title = "All 37 matches" if name == "set1" else "18 matches"
        fig.suptitle(f"Question 3: {title}", fontsize=17, fontweight="bold")
        fig.text(0.5, 0.025, "Shapes aligned by translation, uniform scale, and rotation or reflection.",
                 ha="center", fontsize=10)
        fig.subplots_adjust(left=0.015, right=0.97, top=0.88, bottom=0.10, wspace=0.02)
        fig.savefig(out_dir / f"{name}_shape_comparison.png", dpi=160, facecolor="white")
        plt.close(fig)

        # 3.d, 3.f
        print(name, "singular values =", singular_values)

    # 4.f
    data = data_dir / "statue"
    matches = np.load(data / "matches_subset.npy", allow_pickle=True, encoding="latin1")[0]
    dense_matches = np.load(data / "dense_matches.npy", allow_pickle=True, encoding="latin1")
    fundamentals = np.load(data / "fundamental_matrices.npy", allow_pickle=True, encoding="latin1")[0]

    frames = []
    for pair_matches, F in zip(matches, fundamentals):
        frame = Frame(pair_matches.T, 719.5459, F, estimate_RT_from_E, nonlinear_estimate_3d_point)
        bundle_adjustment(frame)
        frames.append(frame)
    merged = merge_all_frames(frames, nonlinear_estimate_3d_point)

    clouds = []
    for i in range(len(frames) - 1):
        cameras = merged.K @ merged.motion[i:i + 2]
        observations = dense_matches[i].T.reshape(-1, 2, 2)
        clouds.append(np.array([nonlinear_estimate_3d_point(match, cameras) for match in observations]))
    points = np.vstack(clouds)
    points = points[np.isfinite(points).all(axis=1)]

    # Crop the plot around the statue
    lower, upper = np.percentile(points, [0.25, 99.75], axis=0)
    padding = 0.08 * (upper - lower)
    lower -= padding
    upper += padding
    visible = points[np.all((points >= lower) & (points <= upper), axis=1)]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    plot = ax.scatter(*visible.T, c=visible[:, 2], cmap="cividis",
                      depthshade=False, s=0.7, linewidths=0, alpha=0.85)
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
           zlim=(lower[2], upper[2]), xlabel="X", ylabel="Y", title="Statue reconstruction")
    ax.set_zticks([])
    ax.set_box_aspect(upper - lower)
    ax.set_proj_type("ortho")
    ax.view_init(elev=-100, azim=90)
    fig.colorbar(plot, ax=ax, label="Depth (Z)", shrink=0.60, pad=0.04)
    fig.savefig(out_dir / "statue_reconstruction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
