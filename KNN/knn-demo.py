import numpy
from sklearn.neighbors import KNeighborsClassifier


class KNNCommandClassifier(object):
    def __init__(self):
        self.classifier = KNeighborsClassifier()
        self.all_set = self._load_all_commands()
    
    def _load_commands(self, user_index):
        """ Load commands from a user command file
        Read commands in the spacified file. Divide them into 150 groups
        by every 100 commands.
        """
        with open('masquerade-data/User%s' % str(user_index), 'r') as fp:
            _line_count = 1
            commands_list = list()
            _tmp_100_list = list()
            for line in fp.readlines():
                if _line_count % 100 == 0:
                    commands_list.append(_tmp_100_list)
                    _tmp_100_list = []
                else:
                    _tmp_100_list.append(line.strip('\n'))
                _line_count += 1
        return commands_list

    def _load_all_commands(self):
        """ Get a set for all commands of 50 users.
        Read 50 files to get a really big set consisting of all commands.
        @return: set.
        """
        final_set = set()
        for i in range(1, 51):
            with open('masquerade-data/User%s' % str(i), 'r') as fp:
                commands_list = list()
                for line in fp.readlines():
                    commands_list.append(line.strip('\n'))
            final_set = final_set | set(commands_list)
        return final_set

    def _get_feature(self, commands, all_set):
        """ Get a list of features of commands.
        For every groups of commnds in the command list, calculate the 
        features. Get all features together in a list. Hola.
        """
        commands_map = dict()
        for c in commands:
            if c not in commands_map:
                commands_map[c] = 1
            else:
                commands_map[c] += 1
        feature = list()
        for c in all_set:
            if c not in commands_map:
                feature.append(0)
            else:
                feature.append(commands_map[c])
        return feature
    
    def _load_label(self, file_id):
        """ Get labels.
        """
        label_list = list()
        with open('masquerade_summary.txt', 'r') as fp:
            for line in fp.readlines():
                line = line.strip('\n')
                label_list.append(line.split()[file_id])
        return [0] * 50 + label_list

    def train(self, file_id):
        """ Do training.
        """
        feature_list = list()
        commands_list = self._load_commands(file_id)
        label_list = self._load_label(file_id)
        for commands in commands_list:
            feature = self._get_feature(commands, self.all_set)
            feature_list.append(feature)
        self.classifier.fit(feature_list, label_list)
        return 

    def score(self, file_id):
        feature_list = list()
        commands_list = self._load_commands(file_id)
        label_list = self._load_label(file_id)
        for commands in commands_list:
            feature = self._get_feature(commands, self.all_set)
            feature_list.append(feature)
        return self.classifier.score(feature_list, label_list)

    def predict(self, feature_list=None):
        if feature_list is None:
            commands_list = self._load_commands(50)
            feature = self._get_feature(commands_list[0], self.all_set)
            feature_list = [feature]
        return self.classifier.predict(feature_list)


def main():
    knn = KNNCommandClassifier()
    knn.train(6)
    print(knn.score(9))


if __name__ == '__main__':
    main()