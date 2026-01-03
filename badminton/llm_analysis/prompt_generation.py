PG_BASE_PROMPT = """

Identity: You are a video analysis expert. Your know how to find the patterns in descriptions of 
          motion so you can recogmize this mothion in the future. 
          
Task:

    Given: A set of examples for diffeent badminton shot types, each described as a sequence of pose frames encoded as orientation tokens.
    Output: Generate concise descriptions of the examples for each shot type so that they can be used to classify new pose sequences into one of the shot types.to classify new pose sequences into one of the shot types.
    
      
    Input Format:
        { [Shot-Type: [Examples]]}
        Where:          
        Shot-Type: [00_Short_Serve, 05_Drop_Shot, 13_Long_Serve, 14_Smash]
        Examples: A list of example pose descriptions for this shot type. Each example has the format:
            {Position: {FrontCourt, ServeLine, MidCourt, BackCourt},
             Frames: Frame,left_arm,left_leg,left_torso,right_arm,right_leg,right_torso 
                    t0,P_A1_A2,P_A1_A2,P_A1_A2,P_A1_A2,P_A1_A2,P_A1_A2 
                    t1,P_A1_A2,P_A1_A2,P_A1_A2,P_A1_A2,P_A1_A2,P_A1_A2
                    ... 
             }       
                Each row = one video frame
                Each column = one body segment, described by two connected joint orientations.
                Each entry has the form P_<angle1>_<angle2>,
                    where: <angle1> = orientation (in degrees) of the joint closer to the torso 
                        <angle2> = orientation (in degrees) of the joint further from the torso
                        
                Example interpretation: right_arm = P_315.0_225.0 
                    the upper arm (shoulder to elbow) is oriented at 315.0° 
                    and the forearm (elbow to wrist) at 225.0° 
                    angles measured in a consistent global or camera-aligned reference 
                
                Segment Definitions (columns) 
                    left_arm: left shoulder → left elbow → left wrist 
                    left_leg: left hip → left knee → left ankle 
                    left_torso: left shoulder → left hip → left knee 
                    right_arm: right shoulder → right elbow → right wrist 
                    right_leg: right hip → right knee → right ankle 
                    right_torso: right shoulder → right hip → right knee
        
        Attention Constraint For classification, use only: 
            You may consider Position as context.
            Pay attention to how the orientations start (maninly in first few frames) and then evolve over time.
            Not all examples start at the same point in a shot. So find general descriptions that cover the variations.
            Pay close attention to finding the key differences betweens shot types that are similar to each other like 00_Short_Serve and 13_Long_Serve.


        Additional Guidance: (to improve accuracy) Treat angles as cyclic (e.g., 350° close to 10°).

        Use temporal cues: wrist snap, upper-arm swing direction, contact-height proxies from right_arm evolution,
        and torso rotation from right_torso. Use Position as a prior (e.g., ServeLine increases probability of 
        serve types; BackCourt increases probability of clear/smash; FrontCourt increases drop/net/push). 
        If frames are too few or static, still return your best single label with an appropriately low 
        confidence.
        

    
    Output Format: You will output a list of shot and shot description tuples. Each shot description should include:
        Output Format (strict): Return exactly this JSON structure with a single prediction: 
            { ["Shot-Type": "Shot Description" }
        Where: 
            Shot-Type: One of the shot types
            Shot Descriptions: A concise description of the key motion patterns that define this shot type, including:
                - Typical starting orientations for right_arm and left_arm
                - Typical motion patterns for right_arm and left_arm (e.g., fast downward snap, controlled swing)
                - Typical torso motion patterns (e.g., stable, rotation)
                - Typical position on court (e.g., ServeLine, MidCourt)
                - Key differentiators that help distinguish this shot type from others.
                - And any notes on variations within this shot type.
       
"""


PG_INPUT_PROMPT = """
    Input: Generate descriptions for the following examples. 
    
"""