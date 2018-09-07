# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import os

for ind in xrange(1, 6):
    cmd = 'python3 VGG16_features.py --VGG_LAYER_LIMIT=%s' % (ind)
    os.system(cmd)
