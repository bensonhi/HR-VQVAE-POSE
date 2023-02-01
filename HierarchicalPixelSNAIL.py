class HierarchicalPixelSNAIL(pl.LightningModule):

    def __init__(self, n_codes, n_filters, n_res_blocks, n_snail_blocks, n_condition_blocks, key_channels,
                 value_channels):
        super().__init__()
        self.top = PixelSNAIL(attention=True, input_channels=n_codes, n_codes=n_codes,
                              n_filters=n_filters, n_res_blocks=n_res_blocks,
                              n_snail_blocks=n_snail_blocks, key_channels=key_channels,
                              value_channels=value_channels)

        self.condition_top = nn.Sequential(
            nn.ConvTranspose3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(4, 3, 3), stride=(2,1,1), padding=1),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(3, 4, 4), stride=(1,2,2), padding=1),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(3, 3, 3), stride=(1,1,1), padding=1),
        )
        self.condition_bottom = nn.Sequential(
            nn.Conv3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(4, 3, 3), stride=(2,1,1), padding=1),
            nn.ELU(),
            nn.Conv3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(4, 3, 3), stride=(2,1,1), padding=1),
            nn.ELU(),
            nn.Conv3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(4, 3, 3), stride=(2,1,1), padding=1),
            nn.ELU(),
            nn.Conv3d(in_channels=n_codes, out_channels=n_codes, kernel_size=(4, 3, 3), stride=(2,1,1), padding=1),
        )
        self.bottom = PixelSNAIL2D(
            [32, 32],
            512,
            128,
            5,
            2,
            2,
            128,
            attention=False,
            dropout=0.1,
            n_cond_res_block=2,
            cond_res_channel=1024,
        )

        self.criterion = nn.NLLLoss()

    def forward(self, top_code, bot_code):
        condition_top = self.condition_top(top_code)
        condition_bottom = self.condition_bottom(bot_code)
        condition = torch.cat((condition_top, condition_bottom.repeat(1, 1, 18, 1, 1)), dim=1)
        b, s = condtion.shape[0], condition.shape[1]
        condition = rearrange(condition, 'b c s h w -> (b s) c h w')
        bot_code = rearrange(bot_code, 'b c s h w -> (b s) c h w')
        bot_code = self.bottom(bot_code, condition)
        bot_code = rearrange(bot_code, '(b s) c h w -> b c s h w', b=b, s=s)
        
        top_code = self.top(top_code)
        
        return top_code, bot_code

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        in_top, in_bottom = train_batch
        out_top, out_bottom = self.forward(in_top, in_bottom)
        top_loss = self.criterion(out_top, torch.argmax(in_top, dim=1))
        bottom_loss = self.criterion(out_bottom, torch.argmax(in_bottom, dim=1))
        loss = top_loss + bottom_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        in_top, in_bottom = val_batch
        out_top, out_bottom = self.forward((in_top, in_bottom))
        top_loss = self.criterion(out_top, in_top)
        bottom_loss = self.criterion(out_bottom, in_bottom)
        loss = top_loss + bottom_loss
        self.log('val_loss', loss)