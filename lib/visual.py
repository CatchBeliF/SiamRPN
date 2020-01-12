import visdom


class visual:
    def __init__(self, port=8097):
        self.vis = visdom.Visdom(server='http://127.0.0.1', port=port)
        assert self.vis.check_connection()
        self.counter = 0

    def plot_img(self, img, win=1, name='img'):
        self.vis.image(img, win=win, opts={'title': name})

    def plot_imgs(self, img, win=1, name='multi-channel img'):
        self.vis.images(img, win=win, opts={'title': name})

    def plot_heatmap(self, img, win=1, name='img'):
        self.vis.heatmap(img, win=win)
