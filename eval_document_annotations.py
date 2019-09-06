"""Script to evaluate document-level QE as in the WMT19 shared task."""

import argparse
import functools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


class OverlappingSpans(ValueError):
    pass


class Span(object):
    def __init__(self, segment, start, end):
        """A contiguous span of text in a particular segment.
        Args:
            segment: ID of the segment (0-based).
            start: Character-based start position.
            end: The position right after the last character.
        """
        self.segment = segment
        self.start = start
        self.end = end
        assert self.end >= self.start

    def __len__(self):
        """Returns the length of the span in characters.
        Note: by convention, an empty span has length 1."""
        return max(1, self.end - self.start)

    def count_overlap(self, span):
        """Given another span, returns the number of matched characters.
        Zero if the two spans are in different segments.
        Args:
            span: another Span object.
        Returns:
            The number of matched characters.
        """
        if self.segment != span.segment:
            return 0
        start = max(self.start, span.start)
        end = min(self.end, span.end)
        if end >= start:
            if span.start == span.end or self.start == self.end:
                assert start == end
                return 1  # By convention, the overlap with empty spans is 1.
            else:
                return end - start
        else:
            return 0


class Annotation(object):
    def __init__(self, severity=None, category=None, spans=None):
        """An annotation, which has a severity level (minor, major, or critical)
        and consists of one or more non-overlapping spans.

        Args:
            severity: 'minor', 'major', or 'critical'.
            spans: A list of Span objects.
        """
        # make sure that there is no overlap
        spans = sorted(spans, key=lambda span: (span.segment, span.start))
        segment = -1
        for span in spans:
            if span.segment != segment:
                # first span in this segment
                segment = span.segment
                last_end = span.end
            else:
                # second or later span
                if span.start < last_end:
                    raise OverlappingSpans()
                last_end = span.end

        self.severity = severity
        self.category = category
        self.spans = spans

    def __len__(self):
        """Returns the sum of the span lengths (in characters)."""
        return sum([len(span) for span in self.spans])

    def count_overlap(self, annotation, severity_match=None, category_match=None):
        """Given another annotation with the same severity, returns the number
        of matched characters. If the severities are different, the result is
        penalized according to a severity match matrix.

        Args:
            annotation: another Annotation object.
            severity_match: a dictionary of dictionaries containing match
            penalties for severity pairs.
            severity_match: a dictionary of dictionaries containing match
            penalties for category pairs.
        Returns:
            The number of matched characters, possibly penalized by a severity
            mismatch.
        """
        # TODO: Maybe normalize by annotation length (e.g. intersection over
        # union)?
        # Note: since we're summing the matches, this won't work as expected
        # if there are overlapping spans (which we assume there aren't).
        matched = 0
        for span in self.spans:
            for annotation_span in annotation.spans:
                matched += span.count_overlap(annotation_span)

        # Compute exact mathings for severities, categories, both, and
        # top-level category matching.
        severity_exact_matched = matched * (self.severity == annotation.severity)
        category_exact_matched = matched * (self.category == annotation.category)
        both_exact_matched = matched * (
            self.severity == annotation.severity and self.category == annotation.category)
        category_exact_top_matched = matched * (
            top_category(self.category) == top_category(annotation.category))

        # Scale overlap by a coefficient that takes into account mispredictions
        # of the severity. For example, predicting "major" when the error is
        # "critical" gives some partial credit. If None, give zero credit unless
        # the severity is correct.
        severity_discount_matched = matched * severity_match[self.severity][annotation.severity]
        category_discount_matched = matched * category_match(self.category, annotation.category)
        both_discount_matched = (
            matched
            * severity_match[self.severity][annotation.severity]
            * category_match(self.category, annotation.category))

        return (
            matched,
            severity_exact_matched,
            category_exact_matched,
            both_exact_matched,
            category_exact_top_matched,
            severity_discount_matched,
            category_discount_matched,
            both_discount_matched,
        )

    @classmethod
    def from_fields(cls, fields):
        """Creates an Annotation object by loading from a list of string fields.

        Args:
            fields: a list of strings containing annotations information. They
                are:
                - segment_id
                - annotation_start
                - annotation_length
                - severity
                - category

                The first three fields may contain several integers separated by
                whitespaces, in case there are multiple spans.
                The two last fields are ignored.
                Example: "13 13   229 214 7 4     minor fluency"
        """
        segments = list(map(int, fields[0].split(' ')))
        starts = list(map(int, fields[1].split(' ')))
        lengths = list(map(int, fields[2].split(' ')))
        assert len(segments) == len(starts) == len(lengths)
        severity = fields[3]
        category = fields[4]
        spans = [Span(segment, start, start + length)
                 for segment, start, length in zip(segments, starts, lengths)]
        return cls(severity, category, spans)

    @classmethod
    def from_string(cls, line):
        """Creates an Annotation object by loading from a string.
        Args:
            line: tab-separated line containing the annotation information. The
                fields are:
                - document_id
                - segment_id
                - annotation_start
                - annotation_length
                - severity
                - category

                Segment id, annotation start and length may contain several
                integers separated by whitespaces, in case there are multiple
                spans.
                Example: "A0034 13 13   229 214 7 4     minor fluency"
        """
        # Ignore the last two fields.
        fields = line.split('\t')
        assert len(fields) == 6
        return cls.from_fields(fields[1:])

    def to_string(self):
        """Return a string representation of this annotation.

        This is the representation expected in the output file, without notes"""
        segments = []
        starts = []
        lengths = []
        for span in self.spans:
            segments.append(str(span.segment))
            starts.append(str(span.start))
            lengths.append(str(span.end - span.start))

        segment_string = ' '.join(segments)
        start_string = ' '.join(starts)
        length_string = ' '.join(lengths)
        return '\t'.join([segment_string, start_string, length_string,
                          self.severity, self.category])


class Evaluator(object):
    def __init__(self):
        """A document-level QE evaluator."""
        # The severity match matrix will give some credit when the
        # severity is slighted mispredicted ("minor" <> "major" and
        # "major" <> "critical"), but not for extreme mispredictions
        # ("minor" <> "critical").
        self.severity_match = {'minor': {'minor': 1.0,
                                         'major': 0.5,
                                         'critical': 0.0},
                               'major': {'minor': 0.5,
                                         'major': 1.0,
                                         'critical': 0.5},
                               'critical': {'minor': 0.0,
                                            'major': 0.5,
                                            'critical': 1.0}}
        self.category_match = lambda gold, pred: 1 / category_distance(gold, pred)

    def run(self, system, reference, verbose=False):
        """Given system and reference documents, computes the macro-averaged F1
        across all documents.

        Args:
            system: a dictionary mapping names (doc id's) to lists of
                Annotations produced by a QE system.
            reference: a dictionary mapping names (doc id's) to lists of
                reference Annotations.
        Returns:
            The macro-averaged F1 score.
        """
        total_f1 = defaultdict(float)
        annotation_labels = defaultdict(list)
        for doc_id in system:
            # both dicts are defaultdics, returning a empty list if there are no
            # annotations for that doc_id
            reference_annotations = reference[doc_id]
            system_annotations = system[doc_id]

            f1_scores, doc_annotation_labels = self._compare_document(
                system_annotations, reference_annotations)

            if verbose:
                print(doc_id)
                print(' '.join(map(str, f1_scores)))

            total_f1['matching_f1'] += f1_scores[0]
            total_f1['severity_exact_matching_f1'] += f1_scores[1]
            total_f1['category_exact_matching_f1'] += f1_scores[2]
            total_f1['both_exact_matching_f1'] += f1_scores[3]
            total_f1['category_top_matching_f1'] += f1_scores[4]
            total_f1['severity_discount_matching_f1'] += f1_scores[5]
            total_f1['category_discount_matching_f1'] += f1_scores[6]
            total_f1['both_discount_matching_f1'] += f1_scores[7]

            for key, value in doc_annotation_labels.items():
                annotation_labels[key].extend(value)

        for key in total_f1.keys():
            total_f1[key] /= len(system)


        return total_f1, dict(annotation_labels)

    def _compare_document(self, system, reference):
        """Compute the F1 score for a single document, given a system output
        and a reference. This is done by computing a precision according to the
        best possible matching of annotations from the system's perspective,
        and a recall according to the best possible matching of annotations
        from the reference perspective. Gives some partial credit to
        annotations that match with the wrong severity.
        Args:
            system: dictionary mapping doc id's to lists of annotations
            reference: dictionary mapping doc id's to lists of annotations
        Returns:
            The F1 score of a single document.
        """
        num_f1_scores = 8
        all_matched = np.zeros((num_f1_scores, len(system), len(reference)))
        annotation_labels = defaultdict(list)
        for i, system_annotation in enumerate(system):
            for j, reference_annotation in enumerate(reference):
                (
                    matched,
                    severity_exact_matched,
                    category_exact_matched,
                    both_exact_matched,
                    category_exact_top_matched,
                    severity_discount_matched,
                    category_discount_matched,
                    both_discount_matched,
                ) = reference_annotation.count_overlap(
                    system_annotation,
                    severity_match=self.severity_match,
                    category_match=self.category_match
                )
                all_matched[:, i, j] = [
                    matched,
                    severity_exact_matched,
                    category_exact_matched,
                    both_exact_matched,
                    category_exact_top_matched,
                    severity_discount_matched,
                    category_discount_matched,
                    both_discount_matched,
                ]

                annotation_labels['system_severities'].append(
                    system_annotation.severity)
                annotation_labels['system_categories'].append(
                    system_annotation.category)
                annotation_labels['reference_severities'].append(
                    reference_annotation.severity)
                annotation_labels['reference_categories'].append(
                    reference_annotation.category)

        lengths_sys = np.array([len(annotation) for annotation in system])
        lengths_ref = np.array([len(annotation) for annotation in reference])

        if lengths_sys.sum() == 0:
            # no system annotations
            precision = np.ones(num_f1_scores)
        elif lengths_ref.sum() == 0:
            # there were no references
            precision = np.zeros(num_f1_scores)
        else:
            # normalize by annotation length
            precision_by_annotation = all_matched.max(2) / lengths_sys
            precision = precision_by_annotation.mean(1)

        # same as above, for recall now
        if lengths_ref.sum() == 0:
            recall = np.ones(num_f1_scores)
        elif lengths_sys.sum() == 0:
            recall = np.zeros(num_f1_scores)
        else:
            recall_by_annotation = all_matched.max(1) / lengths_ref
            recall = recall_by_annotation.mean(1)

        # simultaneous division of all the values, with division by zero defaulting to 0
        f1 = np.divide(
            2*precision*recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=precision + recall != 0
        )
        assert 0. <= f1.min() and f1.max() <= 1.

        return tuple(f1), dict(annotation_labels)


@functools.lru_cache(maxsize=1024)
def category_distance(gold, pred, level_sep='/'):
    if gold == pred:
        # avoid division by 0 in Evaluator.category_match
        return 1
    else:
        # keep eating items that match till we find the first that does not
        gold = iter(gold.split(level_sep))
        pred = iter(pred.split(level_sep))
        match = True
        while match:
            match = (next(gold) == next(pred))
        # add 2 to account for the level the ate to find the first non-matching level
        return len(list(gold)) + len(list(pred)) + 2


def top_category(category, level_sep='/'):
    return category.split(level_sep)[0]


def load_annotations(file_path):
    """Loads a file containing annotations for multiple documents.

    The file should contain lines with the following format:
    <DOCUMENT ID> <LINES> <SPAN START POSITIONS> <SPAN LENGTHS> <SEVERITY> <CATEGORY>

    Fields are separated by tabs; LINE, SPAN START POSITIONS and SPAN LENGTHS
    can have a list of values separated by white space.

    Args:
        file_path: path to the file.
    Returns:
        a dictionary mapping document id's to a list of annotations.
    """
    annotations = defaultdict(list)

    with open(file_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            fields = line.split('\t')
            doc_id = fields[0]

            try:
                annotation = Annotation.from_fields(fields[1:])
            except OverlappingSpans:
                msg = 'Overlapping spans when reading line %d of file %s '
                msg %= (i, file_path)
                print(msg)
                continue

            annotations[doc_id].append(annotation)

    return annotations


def compute_measures_per_category(true_tags: list, predicted_tags: list, target_tokens: list):
    """Create detailed confusion matrices.

    Code taken from https://gitlab.com/Unbabel/smartcheck/blob/master/metrics/measure.py.
    """
    golden = pd.Series(true_tags)
    predicted = pd.Series(predicted_tags)
    tokens = pd.Series(target_tokens)

    categories_names = list(set(predicted.unique()).union(set(golden.unique())))

    categories_names = list(sorted(categories_names))

    df = pd.DataFrame(columns=['precision', 'recall', 'f1-score', 'support', 'predicted', 'TP', 'FP', 'FN', 'TN'],
                      index=categories_names)
    df.index.name = 'category'

    errors_df = pd.DataFrame(columns=['TP', 'FP', 'FN'], index=categories_names)
    errors_df.index.name = 'category'

    for category in categories_names:
        counter_category = 'OK' if category != 'OK' else 'BAD'

        tags_of_interest = (predicted == category) | (golden == category)
        category_tokens = tokens[tags_of_interest]

        target_tags = golden[tags_of_interest]
        target_tags[target_tags != category] = counter_category

        candidate_tags = predicted[tags_of_interest]
        candidate_tags[candidate_tags != category] = counter_category

        # Confusion matrix
        cm = metrics.confusion_matrix(target_tags.tolist(), candidate_tags.tolist(), labels=[category, counter_category])
        tp = cm[0, 0]  # True positives for category
        fp = cm[1, 0]  # False positives category
        fn = cm[0, 1]  # False negatives category
        tn = cm[1, 1]  # True negatives category (true positives OK)

        precision = tp / (tp + fp + 1e-30)
        recall = tp / (tp + fn + 1e-30)
        df.loc[category] = {
            'precision': precision,
            'recall': recall,
            'f1-score': (2 * precision * recall) / (precision + recall + 1e-30),
            'support': int(tp + fn),
            'predicted': int(tp + fp),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
        }

        # Error analysis
        true_positive_tokens = category_tokens[(target_tags == category) & (candidate_tags == category)].value_counts()
        false_positive_tokens = category_tokens[(target_tags != category) & (candidate_tags == category)].value_counts()
        false_negative_tokens = category_tokens[(target_tags == category) & (candidate_tags != category)].value_counts()
        errors_df.loc[category] = {
            'TP': {'tokens': true_positive_tokens.index.tolist(), 'frequency': true_positive_tokens.values.tolist()},
            'FP': {'tokens': false_positive_tokens.index.tolist(), 'frequency': false_positive_tokens.values.tolist()},
            'FN': {'tokens': false_negative_tokens.index.tolist(), 'frequency': false_negative_tokens.values.tolist()},
        }

    precision = df['TP'].sum() / (df['predicted'].sum() + 1e-30)
    recall = df['TP'].sum() / (df['support'].sum() + 1e-30)
    f1 = (2 * precision * recall) / (precision + recall + 1e-30)
    df.loc['Total'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': df['support'].sum(),
        'predicted': df['predicted'].sum(),
        'TP': df['TP'].sum(),
        'FP': df['FP'].sum(),
        'FN': df['FN'].sum(),
        'TN': df['TN'].sum(),
    }

    return df, errors_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System annotations')
    parser.add_argument('ref', help='Reference annotations')
    parser.add_argument('-v', help='Show score by document',
                        action='store_true', dest='verbose')
    parser.add_argument('-c', help='Save confusion matrix',
                        action='store_true', dest='confusion')
    parser.add_argument('-o', help='Output path for confusion matrices',
                        default='.', dest='output')
    args = parser.parse_args()

    system = load_annotations(args.system)
    reference = load_annotations(args.ref)
    evaluator = Evaluator()
    f1, annotation_labels = evaluator.run(system, reference, args.verbose)
    for key, value in f1.items():
        title = key.replace('_', ' ')
        pretty_title = title[0].upper() + title[1:-2] + title[-2:].upper()
        print('{}: {}'.format(pretty_title, round(value, 4)))

    if args.confusion:
        severities_confusion_df, _ = compute_measures_per_category(
            annotation_labels['system_severities'],
            annotation_labels['reference_severities'],
            annotation_labels['reference_severities'],  # this is just dummy for tokens
        )
        categories_confusion_df, _ = compute_measures_per_category(
            annotation_labels['system_categories'],
            annotation_labels['reference_categories'],
            annotation_labels['reference_categories'],  # this is just dummy for tokens
        )

        output_path = Path(args.output)
        severities_path = output_path / 'severities_confusion.tsv'
        categories_path = output_path / 'categories_confusion.tsv'

        print('Saving confusion matrices to {} and {}'.format(
            severities_path, categories_path))

        severities_confusion_df.to_csv(severities_path, sep='\t')
        categories_confusion_df.to_csv(categories_path, sep='\t')


if __name__ == '__main__':
    main()
