"""
Tests for the reward model components.
"""

import pytest
import numpy as np
from rl.reward_model import (
    RewardModel, SimpleRatingReward, MultiCriteriaReward, 
    LearnedRewardModel, create_reward_model
)


class TestSimpleRatingReward:
    """Test cases for SimpleRatingReward model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = SimpleRatingReward(default_reward=0.3)
        assert model.default_reward == 0.3
        assert len(model.rating_history) == 0
    
    def test_positive_rating(self):
        """Test scoring with positive rating."""
        model = SimpleRatingReward()
        reward = model.score_response(
            "Hello", "Hi there!", {"rating": 1}
        )
        assert reward == 1.0
        assert len(model.rating_history) == 1
    
    def test_negative_rating(self):
        """Test scoring with negative rating."""
        model = SimpleRatingReward()
        reward = model.score_response(
            "Hello", "Bad response", {"rating": 0}
        )
        assert reward == 0.0
        assert len(model.rating_history) == 1
    
    def test_no_feedback(self):
        """Test scoring without feedback."""
        model = SimpleRatingReward(default_reward=0.5)
        reward = model.score_response("Hello", "Hi there!", None)
        assert reward == 0.5
        assert len(model.rating_history) == 0
    
    def test_invalid_rating(self):
        """Test scoring with invalid rating."""
        model = SimpleRatingReward(default_reward=0.3)
        reward = model.score_response(
            "Hello", "Hi there!", {"rating": 2}
        )
        assert reward == 0.3  # Should use default
    
    def test_update_from_feedback(self):
        """Test updating model with feedback data."""
        model = SimpleRatingReward()
        feedback_data = [
            {"user_message": "Test", "response": "Response", "rating": 1},
            {"user_message": "Test2", "response": "Response2", "rating": 0}
        ]
        model.update_from_feedback(feedback_data)
        assert len(model.rating_history) == 2


class TestMultiCriteriaReward:
    """Test cases for MultiCriteriaReward model."""
    
    def test_initialization_default_weights(self):
        """Test initialization with default weights."""
        model = MultiCriteriaReward()
        expected_weights = {
            'relevance': 0.3,
            'helpfulness': 0.3,
            'safety': 0.2,
            'coherence': 0.2
        }
        assert model.criteria_weights == expected_weights
    
    def test_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        custom_weights = {
            'relevance': 0.5,
            'helpfulness': 0.5
        }
        model = MultiCriteriaReward(criteria_weights=custom_weights)
        assert model.criteria_weights == custom_weights
    
    def test_initialization_invalid_weights(self):
        """Test initialization with invalid weights."""
        invalid_weights = {'relevance': 0.3, 'helpfulness': 0.3}  # Sums to 0.6
        with pytest.raises(ValueError):
            MultiCriteriaReward(criteria_weights=invalid_weights)
    
    def test_scoring_with_all_criteria(self):
        """Test scoring with all criteria provided."""
        model = MultiCriteriaReward()
        feedback = {
            'relevance': 1.0,
            'helpfulness': 0.8,
            'safety': 1.0,
            'coherence': 0.9
        }
        reward = model.score_response("Question", "Answer", feedback)
        
        # Expected: 1.0*0.3 + 0.8*0.3 + 1.0*0.2 + 0.9*0.2 = 0.92
        expected = 1.0*0.3 + 0.8*0.3 + 1.0*0.2 + 0.9*0.2
        assert abs(reward - expected) < 1e-6
    
    def test_scoring_with_rating_fallback(self):
        """Test scoring when only overall rating is provided."""
        model = MultiCriteriaReward()
        feedback = {'rating': 0.8}
        reward = model.score_response("Question", "Answer", feedback)
        
        # All criteria should use 0.8, so weighted sum is 0.8
        assert abs(reward - 0.8) < 1e-6
    
    def test_scoring_no_feedback(self):
        """Test scoring without feedback."""
        model = MultiCriteriaReward()
        reward = model.score_response("Question", "Answer", None)
        assert reward == 0.5
    
    def test_update_from_feedback(self):
        """Test updating model with feedback data."""
        model = MultiCriteriaReward()
        feedback_data = [
            {
                'user_message': 'Test',
                'response': 'Response',
                'relevance': 1.0,
                'helpfulness': 0.8
            }
        ]
        model.update_from_feedback(feedback_data)
        assert len(model.feedback_history) == 1


class TestLearnedRewardModel:
    """Test cases for LearnedRewardModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LearnedRewardModel()
        assert model.model is None
        assert len(model.training_data) == 0
    
    def test_scoring_no_model(self):
        """Test scoring when no model is trained."""
        model = LearnedRewardModel()
        reward = model.score_response("Question", "Answer", None)
        assert reward == 0.5  # Default when no model
    
    def test_update_from_feedback(self):
        """Test updating model with feedback data."""
        model = LearnedRewardModel()
        feedback_data = [
            {'user_message': 'Test', 'response': 'Response', 'rating': 1}
        ]
        model.update_from_feedback(feedback_data)
        assert len(model.training_data) == 1


class TestRewardModelFactory:
    """Test cases for the reward model factory function."""
    
    def test_create_simple_model(self):
        """Test creating simple rating model."""
        model = create_reward_model("simple", default_reward=0.4)
        assert isinstance(model, SimpleRatingReward)
        assert model.default_reward == 0.4
    
    def test_create_multi_criteria_model(self):
        """Test creating multi-criteria model."""
        custom_weights = {'relevance': 0.6, 'helpfulness': 0.4}
        model = create_reward_model("multi_criteria", criteria_weights=custom_weights)
        assert isinstance(model, MultiCriteriaReward)
        assert model.criteria_weights == custom_weights
    
    def test_create_learned_model(self):
        """Test creating learned model."""
        model = create_reward_model("learned")
        assert isinstance(model, LearnedRewardModel)
    
    def test_create_invalid_model(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError):
            create_reward_model("invalid_type")


class TestRewardModelInterface:
    """Test that all reward models implement the interface correctly."""
    
    def test_simple_rating_interface(self):
        """Test SimpleRatingReward implements RewardModel interface."""
        model = SimpleRatingReward()
        assert isinstance(model, RewardModel)
        
        # Test abstract methods exist and are callable
        assert hasattr(model, 'score_response')
        assert hasattr(model, 'update_from_feedback')
        assert callable(model.score_response)
        assert callable(model.update_from_feedback)
    
    def test_multi_criteria_interface(self):
        """Test MultiCriteriaReward implements RewardModel interface."""
        model = MultiCriteriaReward()
        assert isinstance(model, RewardModel)
        
        assert hasattr(model, 'score_response')
        assert hasattr(model, 'update_from_feedback')
        assert callable(model.score_response)
        assert callable(model.update_from_feedback)
    
    def test_learned_model_interface(self):
        """Test LearnedRewardModel implements RewardModel interface."""
        model = LearnedRewardModel()
        assert isinstance(model, RewardModel)
        
        assert hasattr(model, 'score_response')
        assert hasattr(model, 'update_from_feedback')
        assert callable(model.score_response)
        assert callable(model.update_from_feedback)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])
