# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F




class DiceLoss(nn.Module):
    """
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5

    def forward(self, logits, labels):
        if len(labels.shape) != len(logits.shape):
            labels = torch.unsqueeze(labels, 1)
        num_classes = logits.shape[1]
        mask = (labels != self.ignore_index)
        logits = logits * mask
        single_label_lists = []
        for c in range(num_classes):
            single_label = (labels == c).int()
            single_label = torch.squeeze(single_label, axis=1)
            single_label_lists.append(single_label)
        labels_one_hot = torch.stack(tuple(single_label_lists), axis=1)
        logits = F.softmax(logits, dim=1)
        labels_one_hot = labels_one_hot.float()
        dims = (0,) + tuple(range(2, labels.ndimension()))
        intersection = torch.sum(logits * labels_one_hot, dims)
        cardinality = torch.sum(logits + labels_one_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss
