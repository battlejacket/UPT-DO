from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class ffsModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            latent,
            decoder,
            conditioner=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        # Re embed
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.input_shape,
        )
        # set static_ctx["dim"]
        if self.conditioner is not None:
            self.static_ctx["dim"] = self.conditioner.dim
        else:
            self.static_ctx["dim"] = latent["kwargs"]["dim"]
        # encoder
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        assert self.encoder.output_shape is not None
        # latent
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs,
        )
        # decoder
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
        )

    @property
    def submodels(self):
        return dict(
            **(dict(conditioner=self.conditioner) if self.conditioner is not None else {}),
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, mesh_pos, query_pos, batch_idx, unbatch_idx, unbatch_select, re):
        outputs = {}

        # encode timestep t
        if self.conditioner is not None:
            condition = self.conditioner(re=re)
        else:
            condition = None  
        
        # encode data
        encoded = self.encoder(mesh_pos=mesh_pos, batch_idx=batch_idx, condition=condition)

        # propagate
        propagated = self.latent(encoded, condition=condition)

        # decode
        x_hat = self.decoder(propagated, query_pos=query_pos, unbatch_idx=unbatch_idx, unbatch_select=unbatch_select, condition=condition)
        outputs["x_hat"] = x_hat

        return outputs
