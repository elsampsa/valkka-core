Definitions:

A Decoder class decodes a frame

A DecoderThread class uses a Decoder and takes care of queueing the frames, mutexes, synchronization, etc.

Details

Decoder subclassing as currently practised:

Decoder (virtual)
    BasicFrame input_frame
    output_frame: defined in subclasses as type depends on decoder
    flush()
    virtual Frame *output() = 0 : returns a reference to output_frame
    virtual bool pull() = 0 # decode!
    isOK() # is the decoder still working?
    
    AVDecoder : Decoder (virtual)
        Audio or video decoding using libav
        AVCodecID, AVPAcket, AVCodecContext, etc. libav paraphernalia initiated at the ctor

        VideoDecoder : AVDecoder
            Video decoding using libav
            AVBitmapFrame out_frame
            implements Frame *output() that returns out_frame
            implements pull(): AVPacket's etc. filled with in_frame, results into out_frame

    AVHwDecoder : Decoder (virtual)
        Audio or video decoding using libav with hardware acceleration
        Takes as an input, the libav AVHWDeviceType
        Like AVDecoder, but creates also AVBufferRef hardware context:
        AVCodecID, AVPAcket, AVCodecContext, AVBufferRef etc. libav paraphernalia initiated at the ctor
        NOTE: most of the ctor code copy-pasted from AVDecoder
        
        HWVideoDecoder : AVHwDecoder
            Hw accelerated video decoding using libav
            Takes as an input, the libav AVHWDeviceType
            Identical (a copy-paste) to VideoDecoder


NOTE: if all the contexes, etc. were _not_ created at ctor but in another method (say, "init"), we could create saner subclass structure.
--> then the DecoderThread::chooseVideoDecoder should call that "init" method just after "new".

DecoderThread classes use the Decoder classes and run the queues, semaphores, etc.

DecoderThread

    virtual Decoder* chooseVideoDecoder(AVCodecID codec_id);
        returns a Decoder instance or NULL of none avail for the codec_id

    virtual Decoder* fallbackVideoDecoder(AVCodecID codec_id);
        returns a second option Decoder instance or NULL of none avail for the codec_id

    run
        - DecoderThread receives SetupFrame(s) and BasicFrames(s)
        - On receiving a SetupFrame, it tries to get a new Decoder by
          calling chooseVideoDecoder
        - It regularly calls Decoder::isOk() -> if that returns false,
          it calls fallbackVideoDecoder






VideoDecoder::pull logic:

current_pixel_format assumed initially AV_PIX_FMT_YUV420P

Then in method pull:

if AV_PIX_FMT_YUV420P --> decode directly to out_frame.av_frame (all good!)

if not, decode to aux_av_frame

--> both refer to av_ref_frame reference pointer

After the fact, check the frame's (av_ref_frame) pix fmt.  If it has changed, set up infra to sws scale the pix fmt into AV_PIX_FMT_YUV420P





