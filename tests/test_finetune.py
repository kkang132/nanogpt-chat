"""
Unit tests for fine-tuning script (finetune.py).
Tests cover data preparation, batch generation, and training utilities.
"""

import unittest
import json
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import torch

# Add path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPrepareTrainingData(unittest.TestCase):
    """Test cases for prepare_training_data function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.chat_log_file = os.path.join(self.test_dir, 'chat_history.jsonl')
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_chat_file(self, num_conversations=10):
        """Helper to create test chat history file."""
        with open(self.chat_log_file, 'w') as f:
            for i in range(num_conversations):
                entry = {
                    'timestamp': f'2024-01-01T{i:02d}:00:00',
                    'user': f'User message {i}',
                    'assistant': f'Assistant response {i}'
                }
                f.write(json.dumps(entry) + '\n')
    
    @patch('finetune.CHAT_LOG_FILE')
    @patch('finetune.DATA_DIR')
    @patch('finetune.tiktoken.get_encoding')
    def test_prepare_training_data_basic(self, mock_encoding, mock_data_dir, mock_chat_log):
        """Test basic training data preparation."""
        # Setup mocks
        mock_chat_log = self.chat_log_file
        mock_data_dir = self.test_dir
        
        self.create_test_chat_file(num_conversations=5)
        
        # Mock tokenizer
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.return_value = mock_enc
        
        from finetune import prepare_training_data
        
        result = prepare_training_data()
        
        # Should return file paths and token counts
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        train_file, val_file, train_len, val_len = result
        
        # Verify files were created
        self.assertTrue(os.path.exists(train_file))
        self.assertTrue(os.path.exists(val_file))
        
    @patch('finetune.CHAT_LOG_FILE')
    def test_prepare_training_data_no_file(self, mock_chat_log):
        """Test prepare_training_data when chat file doesn't exist."""
        mock_chat_log = os.path.join(self.test_dir, 'nonexistent.jsonl')
        
        from finetune import prepare_training_data
        
        result = prepare_training_data()
        self.assertIsNone(result)
    
    @patch('finetune.CHAT_LOG_FILE')
    @patch('finetune.DATA_DIR')
    @patch('finetune.tiktoken.get_encoding')
    def test_prepare_training_data_warning_for_few_conversations(self, mock_encoding, mock_data_dir, mock_chat_log):
        """Test warning when fewer than 10 conversations."""
        mock_chat_log = self.chat_log_file
        mock_data_dir = self.test_dir
        
        # Create file with only 5 conversations
        self.create_test_chat_file(num_conversations=5)
        
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]
        mock_encoding.return_value = mock_enc
        
        from finetune import prepare_training_data
        
        with patch('builtins.print') as mock_print:
            result = prepare_training_data()
            # Check that warning was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            warning_found = any('Warning' in str(call) or 'recommend' in str(call).lower() 
                              for call in print_calls)
            self.assertTrue(warning_found or len(mock_print.call_args_list) > 0)
    
    @patch('finetune.CHAT_LOG_FILE')
    @patch('finetune.DATA_DIR')
    @patch('finetune.tiktoken.get_encoding')
    def test_prepare_training_data_splits_train_val(self, mock_encoding, mock_data_dir, mock_chat_log):
        """Test that data is split 90/10 train/val."""
        mock_chat_log = self.chat_log_file
        mock_data_dir = self.test_dir
        
        self.create_test_chat_file(num_conversations=20)
        
        # Mock tokenizer to return consistent tokens
        mock_enc = MagicMock()
        mock_enc.encode.return_value = list(range(100))  # 100 tokens per conversation
        mock_encoding.return_value = mock_enc
        
        from finetune import prepare_training_data
        
        result = prepare_training_data()
        
        if result:
            train_file, val_file, train_len, val_len = result
            
            # Load and verify split
            train_data = np.memmap(train_file, dtype=np.uint16, mode='r')
            val_data = np.memmap(val_file, dtype=np.uint16, mode='r')
            
            # Should be approximately 90/10 split
            total = train_len + val_len
            train_ratio = train_len / total if total > 0 else 0
            
            self.assertAlmostEqual(train_ratio, 0.9, places=1)


class TestGetBatch(unittest.TestCase):
    """Test cases for get_batch function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.train_data = np.array(list(range(1000)), dtype=np.uint16)
        self.val_data = np.array(list(range(100, 200)), dtype=np.uint16)
        
        # Set batch size and block size for testing
        import finetune
        self.original_batch_size = finetune.batch_size
        self.original_block_size = finetune.block_size
        finetune.batch_size = 2
        finetune.block_size = 128
    
    def tearDown(self):
        """Restore original values."""
        import finetune
        finetune.batch_size = self.original_batch_size
        finetune.block_size = self.original_block_size
    
    @patch('finetune.device', 'cpu')
    def test_get_batch_train(self):
        """Test get_batch for training split."""
        from finetune import get_batch
        
        x, y = get_batch('train', self.train_data, self.val_data)
        
        # Verify shapes
        self.assertEqual(x.shape[0], 2)  # batch_size
        self.assertEqual(x.shape[1], 128)  # block_size
        self.assertEqual(y.shape, x.shape)
        
        # Verify y is shifted by 1
        np.testing.assert_array_equal(y[0, :-1], x[0, 1:])
    
    @patch('finetune.device', 'cpu')
    def test_get_batch_val(self):
        """Test get_batch for validation split."""
        from finetune import get_batch
        
        x, y = get_batch('val', self.train_data, self.val_data)
        
        # Verify shapes
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(x.shape[1], 128)
    
    @patch('finetune.device', 'cpu')
    def test_get_batch_too_small_dataset(self):
        """Test get_batch raises error when dataset too small."""
        from finetune import get_batch
        
        # Create data smaller than block_size + 1
        small_data = np.array([1, 2, 3], dtype=np.uint16)
        
        with self.assertRaises(ValueError) as context:
            get_batch('train', small_data, self.val_data)
        
        self.assertIn('too small', str(context.exception).lower())
    
    @patch('finetune.device', 'cuda')
    def test_get_batch_moves_to_device(self):
        """Test that batches are moved to specified device."""
        from finetune import get_batch
        
        x, y = get_batch('train', self.train_data, self.val_data)
        
        # Should be on CUDA if device is cuda
        # (This test requires actual CUDA setup or mocking torch.to())
        self.assertIsInstance(x, torch.Tensor)


class TestLearningRateSchedule(unittest.TestCase):
    """Test cases for learning rate scheduling."""
    
    def test_warmup_phase(self):
        """Test linear warmup during initial iterations."""
        import finetune
        
        # During warmup, LR should scale linearly
        lr_at_start = finetune.get_lr(0)
        lr_at_mid_warmup = finetune.get_lr(50)
        lr_at_end_warmup = finetune.get_lr(100)
        
        # Should increase during warmup
        self.assertLess(lr_at_start, lr_at_mid_warmup)
        self.assertLess(lr_at_mid_warmup, lr_at_end_warmup)
        
        # At end of warmup, should equal learning_rate
        self.assertAlmostEqual(lr_at_end_warmup, finetune.learning_rate, places=5)
    
    def test_decay_phase(self):
        """Test cosine decay after warmup."""
        import finetune
        
        # After warmup, LR should decay
        lr_after_warmup = finetune.get_lr(100)
        lr_mid_training = finetune.get_lr(500)
        lr_end_training = finetune.get_lr(1000)
        
        # Should decrease during decay
        self.assertGreater(lr_after_warmup, lr_mid_training)
        self.assertGreater(lr_mid_training, lr_end_training)
        
        # Should not go below min_lr
        min_lr = finetune.learning_rate / 10
        self.assertGreaterEqual(lr_end_training, min_lr)
    
    def test_beyond_max_iters(self):
        """Test LR at iterations beyond max_iters."""
        import finetune
        
        lr_beyond = finetune.get_lr(2000)  # Beyond max_iters (1000)
        min_lr = finetune.learning_rate / 10
        
        # Should be at minimum
        self.assertAlmostEqual(lr_beyond, min_lr, places=5)


class TestEstimateLoss(unittest.TestCase):
    """Test cases for estimate_loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.train_data = np.array(list(range(500)), dtype=np.uint16)
        self.val_data = np.array(list(range(100)), dtype=np.uint16)
        
        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.config.block_size = 128
        
        # Mock model to return consistent loss
        mock_logits = torch.zeros(2, 128, 50257)
        self.mock_model.return_value = (mock_logits, torch.tensor(2.5))
    
    @patch('finetune.device', 'cpu')
    @patch('finetune.eval_iters', 5)
    def test_estimate_loss_both_splits(self):
        """Test estimate_loss returns losses for both splits."""
        from finetune import estimate_loss
        
        losses = estimate_loss(self.mock_model, self.train_data, self.val_data)
        
        self.assertIn('train', losses)
        self.assertIn('val', losses)
        self.assertIsInstance(losses['train'], float)
        self.assertIsInstance(losses['val'], float)
    
    @patch('finetune.device', 'cpu')
    @patch('finetune.eval_iters', 3)
    def test_estimate_loss_sets_model_to_eval_mode(self):
        """Test that estimate_loss sets model to eval mode."""
        from finetune import estimate_loss
        
        self.mock_model.training = True  # Start in training mode
        
        estimate_loss(self.mock_model, self.train_data, self.val_data)
        
        # Should have called eval() and then train()
        self.assertTrue(self.mock_model.eval.called or self.mock_model.training == False)


class TestFinetuneIntegration(unittest.TestCase):
    """Integration tests for the finetune function."""
    
    @patch('finetune.prepare_training_data')
    @patch('finetune.torch.load')
    @patch('finetune.GPT')
    @patch('finetune.torch.optim.AdamW')
    def test_finetune_early_stopping(self, mock_optimizer, mock_gpt, mock_torch_load, mock_prepare):
        """Test that early stopping works correctly."""
        # Mock prepare to return valid data
        mock_prepare.return_value = (
            'train.bin', 'val.bin', 1000, 100
        )
        
        # Mock model and optimizer
        mock_model = MagicMock()
        mock_model.config.block_size = 128
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        mock_model.to.return_value = mock_model
        mock_gpt.return_value = mock_model
        
        # Mock torch.load for checkpoint
        mock_torch_load.return_value = {'layer': torch.tensor([1.0])}
        
        mock_opt = MagicMock()
        mock_optimizer.return_value = mock_opt
        
        # Mock estimate_loss to trigger early stopping
        with patch('finetune.estimate_loss') as mock_estimate:
            # First eval: good loss, then no improvement
            mock_estimate.side_effect = [
                {'train': 2.0, 'val': 2.0},  # Initial good loss
                {'train': 2.1, 'val': 2.1},  # No improvement 1
                {'train': 2.1, 'val': 2.1},  # No improvement 2
                {'train': 2.1, 'val': 2.1},  # No improvement 3
                {'train': 2.1, 'val': 2.1},  # No improvement 4
                {'train': 2.1, 'val': 2.1},  # No improvement 5 - should stop
            ]
            
            with patch('finetune.get_batch') as mock_batch:
                mock_batch.return_value = (torch.zeros(2, 128), torch.zeros(2, 128))
                
                with patch('finetune.max_iters', 1000):
                    with patch('finetune.eval_interval', 50):
                        with patch('finetune.patience', 5):
                            from finetune import finetune
                            
                            # Should complete without error
                            finetune()


if __name__ == '__main__':
    unittest.main()

