 

Decoder
    input_frame
    output_frame (in subclasss)
    flush()
    virtual Frame *output() = 0
    virtual bool pull() = 0 # decode!
    isOK() # is the decoder still working
    
AVDecoder : Decoder
    # audio or video using libav
    AVCodecID, AVPAcket, etc. libav paraphernalia
    ctor creates contexts etc.
    # TODO: virtual methods for hw context initialization
    # .. can't do since their virtual methods and would be called in the ctor


HWAVDecoder : Decoder
    # like AVDecoder, but creates a hw context as well

VideoDecoder : AVDecoder
    # video using libav
    # video specific contexes, etc.
    AVBitmapFrame out_frame
    virtual Frame *output()
    virtual bool pull()
    # AVPacket's etc. filled with in_frame


VAAPIDecoder : AVDecoder
    Defines virtual methods for hw context initialization
    AVBitmapFrame out_frame
    # these might look a bit different..?
    virtual Frame *output()
    virtual bool pull()
    



HWVideoDecoder : AVDecoder
    # video using libav
    AVBitmapFrame out_frame
    virtual Frame *output()
    virtual bool pull()



DecoderThread subclasses use Decoder classes
and run the queues, semaphores, etc.

DecoderThread
    virtual Decoder* chooseAudioDecoder(AVCodecID codec_id);
    virtual Decoder* chooseVideoDecoder(AVCodecID codec_id);
    virtual Decoder* fallbackAudioDecoder(AVCodecID codec_id);
    virtual Decoder* fallbackVideoDecoder(AVCodecID codec_id);




