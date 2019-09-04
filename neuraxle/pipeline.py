"""
Neuraxle's Pipeline Classes
====================================
This is the core of Neuraxle's pipelines. You can chain steps to call them one after an other.

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
from abc import ABC, abstractmethod
from typing import Any, Tuple, List

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList, ResumableStepMixin
from neuraxle.checkpoints import BaseCheckpointStep


class DataObject:
    def __init__(self, i, x):
        self.i = i
        self.x = x

        def __hash__(self):
            return hash((self.i, self.x))


class BasePipeline(TruncableSteps, ABC):

    def __init__(self, steps: NamedTupleList):
        BaseStep.__init__(self)
        TruncableSteps.__init__(self, steps)

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BasePipeline':
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data_inputs):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BasePipeline', Any):
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform_processed_outputs(self, data_inputs) -> Any:
        raise NotImplementedError()

    def inverse_transform(self, processed_outputs):
        if self.transform != self.inverse_transform:
            raise BrokenPipeError("Don't call inverse_transform on a pipeline before having mutated it inversely or "
                                  "before having called the `.reverse()` or `reversed(.)` on it.")

        return self.inverse_transform_processed_outputs(processed_outputs)


class Pipeline(BasePipeline, BaseStep, ResumableStepMixin):
    """
    Fits and transform steps after latest checkpoint
    """

    def __init__(self, steps: NamedTupleList):
        super().__init__(steps)

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        new_self, data_inputs = self.fit_transform_steps(data_inputs, expected_outputs)
        processed_outputs = data_inputs

        return new_self, processed_outputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        new_self, _ = self.fit_transform_steps(data_inputs, expected_outputs)

        return new_self

    def fit_transform_steps(self, data_inputs, expected_outputs):
        """
        After loading the last checkpoint, fit transform each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        steps_left_to_do, data_inputs = self.read_checkpoint(data_inputs)

        # TODO: perhaps do this? to push the ifs into a factory and have no ifs in the loop below.
        # decorated_steps_left_to_do: List[PipelineStepHandler] = PipelineStepHandlerFactory.decorate(steps_left_to_do)

        #dos = DataObjectContainer([DataObject(di) for di in data_inputs])
        dos = DataObjectContainer(data_inputs, HasherByIndex())

        last_checkpoint_step = 0
        new_steps_as_tuple: NamedTupleList = []
        for step_name, step in steps_left_to_do:

            # TODO: do just this instead:
            step, data_inputs, expected_outputs, dos = step.handle(data_inputs, expected_outputs, dos, self)
            # end of todo.

            # TODO Other idea:
            if is_currently_packed:
                step, dos = step.handle_packed(
                    dos=dos, truncable=self)
            else:
                step, data_inputs, expected_outputs = step.handle_unpacked(
                    data_inputs=data_inputs, expected_outputs=expected_outputs)
            if transition_to_unpacked:
                data_inputs, expected_outputs = dos.unpack()
            elif transition_to_packed:
                dos.pack(data_inputs, expected_outputs)
            # end of todo other idea.

            # TODO: other idea:
            if isinstance(step, Hasher):
                dos.set_new_hasher(step)
            elif isinstance(step, BaseCheckpointStep):
                to_hash: List[BaseStep] = self[last_checkpoint_step:step_name]
                dos.repack(these_data_inputs, to_hash)
                step, dos = step.fit_transform(dos, expected_outputs)
                these_data_inputs = dos.unpack()
                last_checkpoint_step = step
            else:
                step, these_data_inputs = step.fit_transform(these_data_inputs, expected_outputs)
            # End of todo other idea.

            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = self.steps_as_tuple[:len(self.steps_as_tuple) - len(steps_left_to_do)] + \
                              new_steps_as_tuple

        return self, dos.unpack()

    def tmptest(self):

        # TODO: this is a unit test to finish. Move to test file.

        p = Pipeline([
            # Pack
            HasherByIndex(),
            # Unpack
            SomeStep(),
            SomeStep(),
            SomeStep(),
            SomeStep(),
            SomeStep(),
            # Pack
            HasherByDataHash(),
            Unpack(),
            SomeStep(),
            # Pack
            HasherByIndex(),
            Checkpoint(),
            # Unpack
            SomeStep(),
        ])


    def transform(self, data_inputs):
        """
        After loading the last checkpoint, transform each pipeline steps
        :param data_inputs: the data input to fit on
        :return: transformed data inputs
        """
        steps_left_to_do, data_inputs = self.read_checkpoint(data_inputs)
        for step_name, step in steps_left_to_do:
            data_inputs = step.transform(data_inputs)

        return data_inputs

    def inverse_transform_processed_outputs(self, processed_outputs) -> Any:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs
        :param processed_outputs: the forward transformed data input
        :return: backward transformed processed outputs
        """
        for step_name, step in list(reversed(self.steps_as_tuple)):
            processed_outputs = step.transform(processed_outputs)
        return processed_outputs

    def read_checkpoint(self, data_inputs) -> Tuple[list, Any]:
        new_data_inputs = data_inputs
        new_starting_step_index = self.find_starting_step_index(data_inputs)

        step = self.steps_as_tuple[new_starting_step_index]
        if isinstance(step, BaseCheckpointStep):
            checkpoint = step.read_checkpoint(data_inputs)
            new_data_inputs = checkpoint

        return self.steps_as_tuple[new_starting_step_index:], new_data_inputs

    def find_starting_step_index(self, data_inputs) -> int:
        """
        Find the starting step index that has the latest checkpoint (or 0 if there is no checkpoint)
        :param data_inputs: the data input to fit on
        :return: index, (data_inputs, expected_outputs) tuple for the starting step data inputs-outputs
         and the starting step index
        """
        # TODO: return list of steps directly with __getitem__()
        for index, (step_name, step) in enumerate(reversed(self.steps_as_tuple)):
            if isinstance(step, ResumableStepMixin) and step.should_resume(data_inputs):
                return len(self.steps_as_tuple) - index - 1
        return 0

    def should_resume(self, data_inputs) -> bool:
        """
        Return True if the pipeline has a step that can be resumed (another pipeline or a checkpoint step).
        :param data_inputs: the data input to fit on
        :return: boolean
        """
        for index, (step_name, step) in enumerate(reversed(self.steps_as_tuple)):
            if isinstance(step, ResumableStepMixin) and step.should_resume(data_inputs):
                return True
        return False
