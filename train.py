# training script used in "Physics-assisted Generative Adversarial Network"
# written and maintained by Zhen Guo
# =============================================================================

import sys

# this is for batched training
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

alphas = [1, 1/2, 1/4, 1/8, 1/16, 1/32]

noise_idx = 500

alpha = alphas[my_task_id-1]

print('alpha:', alpha)

loss_init_mean = []
loss_init_std = []
loss_mean = []
loss_std = []

from tqdm import tqdm

for n in tqdm(range(0, 10)):
    
    decoder_optimizer = optimizers.Adam(lr=1e-4, beta_1=0, beta_2=0.9, epsilon=1e-08, decay=0)
    discriminator_optimizer = optimizers.Adam(lr=4e-4, beta_1=0, beta_2=0.9, epsilon=1e-08, decay=0)

    checkpoint_dir = './GAN_training_checkpoints' + '-alpha-' + str(alpha)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=decoder_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     decoder=decoder,
                                     discriminator=discriminator)
    
    @tf.function
    def train_step(recon, images, epoch):
        for _ in range(1):
            with tf.GradientTape() as disc_tape:

                decoded = decoder(recon, training=False)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(decoded, training=True)

                disc_loss = d_hinge_loss(real_output, fake_output)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))


        for _ in range(4):
            with tf.GradientTape() as decode_tape:

                decoded = decoder(recon, training=True)

                #real_output = discriminator(images1, training=True)
                fake_output = discriminator(decoded, training=False)

                expected_loss = npcc(images, decoded)
                gen_loss = alpha * g_hinge_loss(fake_output)
                gen_loss_tot = expected_loss + gen_loss

            gradients_of_decoder = decode_tape.gradient(gen_loss_tot, decoder.trainable_weights)

            decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_weights))

        return gen_loss, gen_loss_tot, expected_loss, disc_loss

    def Callback_EarlyStopping(LossList, min_delta=0.02, patience=20):
        #No early stopping for 2*patience epochs 
        if len(LossList)//patience < 2 :
            return False
        #Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
        mean_recent = np.mean(LossList[::-1][:patience]) #last
        #you can use relative or absolute change
        delta_abs = np.abs(mean_recent - mean_previous) #abs change
        delta_abs = np.abs(delta_abs / mean_previous)  # relative change
        if delta_abs < min_delta :
            print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
            print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
            return True
        else:
            return False

    def train(dataset, test_dataset, epochs):
        gen_loss_list = []
        pure_gen_loss = []
        expected_loss_list = []
        disc_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            start = time.time()

            for recon_batch, image_batch in dataset:
                gen_loss, gen_loss_tot, expected_loss, disc_loss = train_step(recon_batch, image_batch, epoch)
                expected_loss_list.append(expected_loss)
                gen_loss_list.append(gen_loss_tot)
                disc_loss_list.append(disc_loss)
                pure_gen_loss.append(gen_loss)

            #for patterns, image in testing_dataset:
            #    display.clear_output(wait=True)
            #    generate_and_save_images(decoder,
            #                     epoch + 1,
            #                    patterns)
            val_temp = []
            for recon_batch, image_batch in test_dataset:
                decoded = decoder.predict(recon_batch, batch_size=batch,verbose=0)
                val_loss = npcc(image_batch, decoded)
                val_temp.append(val_loss)
            val_loss_list.append(np.mean(val_temp))

            # Save the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            stopEarly = Callback_EarlyStopping(val_loss_list, min_delta=0.001, patience=25)
            if stopEarly:
                print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
                print("Terminating training ")
                break

            if (epoch + 1) % 5 == 0:
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                print ('expected_loss = ' + str(expected_loss.numpy()))
                print ('val_loss = ' + str(val_loss.numpy()))
                print ('gen_loss = ' + str(gen_loss_tot.numpy()))
                print ('disc_loss = ' + str(disc_loss.numpy()))
                print ('pure_gen_loss = ' + str(gen_loss.numpy()))
        return gen_loss_list, expected_loss_list, disc_loss_list
    
    batch = 25
    decoder = create_generator(batch_size=batch)
    decoder.summary()

    discriminator = make_discriminator_model(batch_size=batch)
    discriminator.summary()
    
    ground_truth_ic = np.load('p=0.5-gd.npy')
            
    recon_ic = np.load('p=0.5-' + str(n) + '.npy')
    
    new_ground_truth_ic = ground_truth_ic.astype(np.float32)
    new_recon_ic = recon_ic.astype(np.float32)

    print(np.shape(new_ground_truth_ic))
    print(np.shape(new_recon_ic))
                
    loss_list = []

    for recon, truth in zip(new_recon_ic, new_ground_truth_ic):
        loss = npcc(truth, recon)
        loss_list.append(loss)
    
    loss_init_mean.append(np.mean(loss_list))
    loss_init_std.append(np.std(loss_list))
    
    
    recon_dataset = tf.data.Dataset.from_tensor_slices(new_recon_ic[:1800])
    images_dataset = tf.data.Dataset.from_tensor_slices(new_ground_truth_ic[:1800])

    testing_recon = tf.data.Dataset.from_tensor_slices(new_recon_ic[1800:])
    testing_images = tf.data.Dataset.from_tensor_slices(new_ground_truth_ic[1800:])

    train_dataset = tf.data.Dataset.zip((recon_dataset, 
                                        images_dataset)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(batch)

    testing_dataset = tf.data.Dataset.zip((testing_recon, testing_images)).batch(batch)

    gen_loss_list, expected_loss_list, disc_loss_list = train(train_dataset, testing_dataset, epoch)
    
    #decoder.save('GAN-model-autoencode3D-3D-n-' + str(n) + '-alpha-' + str(alpha) + '.h5')
    
    decoded = decoder.predict(new_recon_ic[1800:], batch_size=batch,verbose=1)
    
    np.save('autoencode3D-idx-' + str(noise_idx) + '-n-' + str(n) + '-alpha-' + str(alpha), decoded)
    
    loss_list = []

    for recon, truth in tqdm(zip(decoded, new_ground_truth_ic[1800:])):
        loss = npcc(truth, recon)
        loss_list.append(loss)
        
    loss_mean.append(np.mean(loss_list))
    loss_std.append(np.std(loss_list))

print(loss_init_mean, loss_mean)
