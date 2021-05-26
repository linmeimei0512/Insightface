import sys
import os
import os.path as osp
import cv2
import numpy as np
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.mask_renderer import face3d
from tools.mask_renderer.face3d import mesh
from tools.mask_renderer.face3d.morphable_model import MorphabelModel
from tools.mask_renderer.image_3d68 import Handler


'''
########################################
Mask Renderer
########################################
'''
class MaskRenderer:
    mask_image_path = os.path.join(os.path.dirname(__file__), 'mask_image/mask_04.png')
    mask_image = None

    face3d_model_path = './assets_mask'

    def __init__(self, use_face3d=False, mask_image_path=None):
        if mask_image_path is not None:
            self.mask_image_path = mask_image_path
        self.use_face3d = use_face3d

        print('\n********** Mask Renderer **********')
        print('use face3d: ' + str(self.use_face3d))
        print('mask image path: ' + str(self.mask_image_path))

        if use_face3d:
            self.init_face3d_mask_renderer()

        self.init_mask_image()
        print('***********************************')


    # ================================================
    # Initialize mask image
    # ================================================
    def init_mask_image(self, positions=[0.1, 0.33, 0.9, 0.7]):
        if not os.path.exists(self.mask_image_path):
            print('Initialize mask image failed. Mask image \'' + str(self.mask_image_path) + '\' is not exist.')
            return

        if self.use_face3d:
            image = cv2.imread(self.mask_image_path)
            self.uv_mask_image = self.generate_mask_uv(image, positions)
        else:
            image = cv2.imread(self.mask_image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (90, 55))

            add_alpha = np.zeros((112, 112, 4), dtype=np.uint8)
            add_alpha[57:112, 11:101] = image

            self.mask_image = add_alpha
        print('Initialize mask image success.')


    # ================================================
    # Render mask
    # ================================================
    def render(self, image, params=None):
        # use face3d
        if self.use_face3d:
            return self.render_for_face3d(image, params)

        # use opencv
        else:
            if self.mask_image is None:
                return

            image = image.copy()

            b, g, r, a = cv2.split(self.mask_image)
            foreground = cv2.merge((b, g, r))

            alpha = cv2.merge((a, a, a))
            alpha = alpha.astype(float) / 255

            foreground = foreground.astype(float)
            background = image.astype(float)

            foreground = cv2.multiply(alpha, foreground)
            background = cv2.multiply(1 - alpha, background)

            outImage = foreground + background
            outImage = outImage.astype(np.uint8)

            return outImage


    # ================================================
    # Initialize face3d mask renderer
    # ================================================
    def init_face3d_mask_renderer(self):
        self.bfm = MorphabelModel(osp.join(self.face3d_model_path, 'BFM.mat'))
        self.index_ind = self.bfm.kpt_ind

        uv_coords = face3d.morphable_model.load.load_uv_coords(osp.join(self.face3d_model_path, 'BFM_UV.mat'))
        self.uv_size = (224, 224)

        self.mask_stxr = 0.1
        self.mask_styr = 0.33
        self.mask_etxr = 0.9
        self.mask_etyr = 0.7
        self.tex_h, self.tex_w, self.tex_c = self.uv_size[1], self.uv_size[0], 3
        texcoord = np.zeros_like(uv_coords)
        texcoord[:, 0] = uv_coords[:, 0] * (self.tex_h - 1)
        texcoord[:, 1] = uv_coords[:, 1] * (self.tex_w - 1)
        texcoord[:, 1] = self.tex_w - texcoord[:, 1] - 1
        self.texcoord = np.hstack((texcoord, np.zeros((texcoord.shape[0], 1))))
        self.X_ind = self.bfm.kpt_ind

        self.if3d68_handler = Handler(osp.join(self.face3d_model_path, 'if1k3d68'), 0, 192, ctx_id=-1)

    # ================================================
    # Transform
    # ================================================
    def transform(self, shape3D, R):
        s = 1.0
        shape3D[:2, :] = shape3D[:2, :]
        shape3D = s * np.dot(R, shape3D)
        return shape3D

    # ================================================
    # Preprocess
    # ================================================
    def preprocess(self, vertices, w, h):
        R1 = mesh.transform.angle2matrix([0, 180, 180])
        t = np.array([-w // 2, -h // 2, 0])
        vertices = vertices.T
        vertices += t
        vertices = self.transform(vertices.T, R1).T
        return vertices

    # ================================================
    # Project to 2d
    # ================================================
    def project_to_2d(self, vertices, s, angles, t):
        transformed_vertices = self.bfm.transform(vertices, s, angles, t)
        projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection
        return projected_vertices[self.bfm.kpt_ind, :2]

    # ================================================
    # Generate mask uv
    # ================================================
    def generate_mask_uv(self, mask, positions):
        uv_size = (self.uv_size[1], self.uv_size[0], 3)
        h, w, c = uv_size
        uv = np.zeros(shape=(self.uv_size[1], self.uv_size[0], 3), dtype=np.uint8)
        stxr, styr = positions[0], positions[1]
        etxr, etyr = positions[2], positions[3]
        stx, sty = int(w * stxr), int(h * styr)
        etx, ety = int(w * etxr), int(h * etyr)
        height = ety - sty
        width = etx - stx
        mask = cv2.resize(mask, (width, height))
        uv[sty:ety, stx:etx] = mask
        return uv

    # ================================================
    # Params to vertices
    # ================================================
    def params_to_vertices(self, params, H, W):
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t  = params
        fitted_vertices = self.bfm.generate_vertices(fitted_sp, fitted_ep)
        transformed_vertices = self.bfm.transform(fitted_vertices, fitted_s, fitted_angles,
                                                  fitted_t)
        transformed_vertices = self.preprocess(transformed_vertices.T, W, H)
        image_vertices = mesh.transform.to_image(transformed_vertices, H, W)
        return image_vertices


    # ================================================
    # Build params
    # ================================================
    def build_params(self, face_image):

        landmark = self.if3d68_handler.get(face_image)[:,:2]
        # print(landmark.shape, landmark)
        #print(landmark.shape, landmark.dtype)
        if landmark is None:
            return None #face not found
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = self.bfm.fit(landmark, self.X_ind, max_iter = 3)
        return [fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t]


    # ================================================
    # Render mask for face3d
    # ================================================
    def render_for_face3d(self, face_image, params, auto_blend=True):
        h, w, c = face_image.shape
        image_vertices = self.params_to_vertices(params, h, w)

        output = (1 - mesh.render.render_texture(image_vertices, self.bfm.full_triangles, self.uv_mask_image, self.texcoord, self.bfm.full_triangles, h, w)) * 255
        output = output.astype(np.uint8)

        if auto_blend:
            mask_bd = (output==255).astype(np.uint8)
            final = face_image*mask_bd + (1-mask_bd)*output
            return final

        return output


'''
=============================
Default
=============================
'''
mask_image_path = 'mask_image/mask_04.png'
image_path = '../../images/Tom_Hanks_01.jpg'
use_face3d = True
save_image_path = '../../images/Tom_Hanks_01(mask).jpg'
show_image = True


'''
=============================
Main
=============================
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mask Renderer')
    parser.add_argument('--mask_image_path', type=str, default=mask_image_path, help='where is the mask photo.')
    parser.add_argument('--image_path', type=str, default=image_path, help='where is the photo that need render mask.')
    parser.add_argument('--use_face3d', action='store_true', default=False, help='use face3d.')
    parser.add_argument('--save_image_path', type=str, default=save_image_path, help='where to save the photo that rendered mask.')
    parser.add_argument('--show_image', type=bool, default=show_image, help='show the photo that rendered mask. default: True.')
    args = parser.parse_args()

    mask_image_path = args.mask_image_path
    image_path = args.image_path
    use_face3d = args.use_face3d
    save_image_path = args.save_image_path
    show_image = args.show_image

    # initialize mask renderer
    mask_renderer = MaskRenderer(use_face3d=use_face3d, mask_image_path=mask_image_path)

    if not os.path.exists(image_path):
        print('Photo \'' + str(image_path) + '\' is not exist.')
        sys.exit()

    image = cv2.imread(image_path)

    if use_face3d:
        start_time = time.time()
        params = mask_renderer.build_params(image)
        print('\nDetect face cost time: ' + str(time.time() - start_time))

        start_time = time.time()
        mask_out = mask_renderer.render(image, params)
        print('Render mask cost time: ' + str(time.time() - start_time))

    else:
        start_time = time.time()
        mask_out = mask_renderer.render(image)
        print('Render mask cost time: ' + str(time.time() - start_time))

    # save photo that rendered mask
    if save_image_path != '':
        cv2.imwrite(save_image_path, mask_out)

    # show photo that rendered mask
    if show_image:
        cv2.imshow('', mask_out)
        cv2.waitKey(0)

