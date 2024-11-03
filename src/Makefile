
# -------------------
# train models
# -------------------
run_training_unconditional_gan:
	 python3 train_unconditional_gan.py

#run_training_conditional_gan:
	 python3 train_conditional_gan.py

# -------------------
# train classificatot in sinthetic DS
# -------------------
run_classification:
	 python3 classificationFakeDS.py \
        --checkpoint cond_generator.epoch_95.pth \
        --logs logs_sgd

# -------------------
# plot loss function
# -------------------
#run_plt_loss_conditional:
run_plt_loss:

	python3 plt_loss.py \
        --fname loss_conditional_gan \
        --type_gan conditional \
        --logs logs_adam_05
        #logs_sgd

#run_plt_loss_evolution_conditional:
	python3 plt_loss_evolution.py \
        --fname evolution_loss_conditional_gan \
        --type_gan conditional \
        --logs logs_adam_05
        #logs_sgd

#run_plt_loss_unconditional:
	python3 plt_loss.py \
        --fname loss_unconditional_gan \
        --type_gan unconditional \
        --logs logs_adam_05
        #logs_sgd

#run_plt_loss_evolution_unconditional:
	python3 plt_loss_evolution.py \
        --fname evolution_loss_unconditional_gan \
        --type_gan unconditional \
        --logs logs_adam_05
        #logs_sgd

run_plt_transition_numbers:
	python3 transitionNumbers.py \
	    --checkpoint cond_generator.epoch_95.pth \
        --logs logs_adam_05 \
        --x 2 \
        --y 1
