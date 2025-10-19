import torch
from src.methods.abstract_meta_learner import AbstractMetaLearner, AbsMetaLearnerR
from src.methods.utils import euclidean_dist
from torchvision import transforms


class ProtoNet(AbsMetaLearnerR):  # switch to an resized AbstractMetaLearner
    def set_forward(self, support_images, support_labels, query_images):
        """
        Overwrites method set_forward in AbstractMetaLearner.
        """
        # print(support_images.shape, query_images.shape)
        if support_images.shape[0] > 350:
            split_times = support_images.shape[0] // 100
            si_size = support_images.shape[0] // split_times
            qi_size = query_images.shape[0] // split_times
            z_support = []
            z_query = []
            for i in range(split_times - 1):
                z_support_temp, z_query_temp = self.extract_features(
                    support_images[i * si_size : (i + 1) * si_size],
                    query_images[i * qi_size : (i + 1) * qi_size],
                )
                z_support.append(z_support_temp)
                z_query.append(z_query_temp)
            z_support_temp, z_query_temp = self.extract_features(
                support_images[(i + 1) * si_size :],
                query_images[(i + 1) * qi_size :],
            )
            z_support.append(z_support_temp)
            z_query.append(z_query_temp)
            z_support = torch.cat(z_support, dim=0)
            z_query = torch.cat(z_query, dim=0)
        else:
            z_support, z_query = self.extract_features(support_images, query_images)

        # If a transportation method in the feature space has been defined, use it
        if self.transportation_module:
            # Prototype i is the mean of all instances of features corresponding to labels == i
            if len(support_labels[support_labels == 0]) > 1:  # if more than 1-shot
                for label in torch.unique(support_labels):  # Inner-OT
                    z_class = z_support[support_labels == label]
                    z_proto_0 = z_class.mean(0).unsqueeze(0)
                    # Closest instance to the mean as the new prototype
                    z_distance = euclidean_dist(z_proto_0, z_class)
                    z_proto_0 = z_class[torch.argmin(z_distance)].unsqueeze(0)
                    z_support[support_labels == label] = self.transportation_module(
                        z_proto_0, z_class
                    )[0]
            z_support, z_query = self.transportation_module(z_support, z_query)

        z_proto = self.get_prototypes(z_support, support_labels)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores
