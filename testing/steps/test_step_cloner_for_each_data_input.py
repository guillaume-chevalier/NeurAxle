import numpy as np

from neuraxle.hyperparams.distributions import Boolean
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import StepClonerForEachDataInput
from neuraxle.steps.misc import FitCallbackStep, TapeCallbackFunction
from neuraxle.steps.numpy import MultiplyByN

HYPE_SPACE = HyperparameterSpace({
    "a__test": Boolean()
})

HYPE_SAMPLE = HyperparameterSamples({
    "a__test": True
})


def test_step_cloner_should_transform():
    tape = TapeCallbackFunction()
    p = StepClonerForEachDataInput(
            Pipeline([
                FitCallbackStep(tape),
                MultiplyByN(2)
            ])
        )
    data_inputs = _create_data((2, 2))

    processed_outputs = p.transform(data_inputs)

    assert isinstance(p.steps[0], Pipeline)
    assert isinstance(p.steps[1], Pipeline)
    assert np.array_equal(processed_outputs, data_inputs * 2)


def test_step_cloner_should_fit_transform():
    # Given
    tape = TapeCallbackFunction()
    p = StepClonerForEachDataInput(
            Pipeline([
                FitCallbackStep(tape),
                MultiplyByN(2)
            ])
        )
    data_inputs = _create_data((2, 2))
    expected_outputs = _create_data((2, 2))

    # When
    p, processed_outputs = p.fit_transform(data_inputs, expected_outputs)

    # Then
    assert isinstance(p.steps[0], Pipeline)
    assert np.array_equal(p.steps[0][0].callback_function.data[0][0], data_inputs[0])
    assert np.array_equal(p.steps[0][0].callback_function.data[0][1], expected_outputs[0])

    assert isinstance(p.steps[1], Pipeline)
    assert np.array_equal(p.steps[1][0].callback_function.data[0][0], data_inputs[1])
    assert np.array_equal(p.steps[1][0].callback_function.data[0][1], expected_outputs[1])

    assert np.array_equal(processed_outputs, data_inputs * 2)


def test_step_cloner_should_inverse_transform():
    tape = TapeCallbackFunction()
    p = StepClonerForEachDataInput(
            Pipeline([
                FitCallbackStep(tape),
                MultiplyByN(2)
            ])
        )
    data_inputs = _create_data((2, 2))
    expected_outputs = _create_data((2, 2))

    p, processed_outputs = p.fit_transform(data_inputs, expected_outputs)
    p = p.reverse()

    assert np.array_equal(processed_outputs, data_inputs * 2)
    inverse_processed_outputs = p.inverse_transform(processed_outputs)
    assert np.array_equal(np.array(inverse_processed_outputs), np.array(data_inputs))


def _create_data(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    return data_inputs
