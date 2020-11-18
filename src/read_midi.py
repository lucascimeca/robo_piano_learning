import os
import mido
#from roll import MidiFile
from mido import Message, MidiFile, MidiTrack, MetaMessage
from mido.frozen import freeze_message, thaw_message
import time

MIDO_DEFAULT_INPUT = 'KAWAI USB MIDI 1'
TICKS_PER_BEAT = 480
TEMPO = 500000

# """ Calibration MIDI script """
# print("Perform Calibration")
piano_keys = {}
# x, y = 0, 0
# t0 = time.time()
# with mido.open_input() as port:
#     for msg in port:
#         if time.time() - t0 < 5.0:
#             print("idle time: ", time.time() - t0)
#             #print(msg)
#             if msg.type == "note_on":
#                 piano_keys[msg.note] = (x, y)
#                 y += 0.02367
#                 print(piano_keys)
#             t0 = time.time()
#         # idle time > 10s, then close port
#         else:
#             print("closing port as no messages were sent for 10 seconds")
#             break
# print("Calibration Complete")


""" Demonstration Save File script """
print("Play Demo")
audio_file = os.path.join(os.getcwd(), r"melody_try.mid")
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.append(MetaMessage("set_tempo", tempo=500000))

with mido.open_input() as port:
    t0 = time.time()
    for msg in port:
        if time.time() - t0 < 20.0:
            msg.time = int(mido.second2tick(time.time() - t0,TICKS_PER_BEAT,TEMPO))
            print(msg.time)
            track.append(msg)
            t0 = time.time()
        else:
            mid.save(audio_file)
            print("Demo Ended")
            break

# """Get position action sequence from midi demo file"""
# print("Generating action list")
# mid = MidiFile(r"C:\Users\chery\Desktop\Workspace\new_song1.mid")
#
# for i, track in enumerate(mid.tracks):
#     for msg in track:
#         print(msg)
#
# action_parameters = []
# #action_parameters holds an array of rows containing [key, key_coord,
# #on_velocity, hold_time, off_velocity, wait_time]
# for i, track in enumerate(mid.tracks):
#     for msg in track:
#         if msg.type == "note_on":
#             on_note = msg.note
#             on_velocity = msg.velocity
#             wait_time = mido.tick2second(msg.time, TICKS_PER_BEAT, TEMPO)
#         elif msg.type == "note_off":
#             off_note = msg.note
#             off_velocity = msg.velocity
#             hold_time = mido.tick2second(msg.time, TICKS_PER_BEAT, TEMPO)
#             if off_note == on_note:
#                 parameter = [piano_keys[off_note], on_velocity, hold_time, off_velocity, wait_time]
#                 action_parameters.append(parameter)
#             else:
#                 print("overlapping notes, can't handle")
#             if len(action_parameters) > 1:
#                 action_parameters[len(action_parameters)-2][4] = wait_time
# action_parameters[len(action_parameters)-1][4] = 0
#
# print("Actions generated:")
# print("[key_coord, on_velocity, hold_time, off_velocity, wait_time]")
# for action in action_parameters:
#     print(action)
# #print(action_parameters)





#        if msg.type == "note_on":
#            track.append(Message("note_on", note=msg.note, 
#                                 velocity=msg.velocity, 
#                                 time=mido.second2tick(time.time()-t0,
#                                                       TICKS_PER_BEAT,TEMPO))
#        elif msg.type == "note_off":
#            track.append(Message("note_off", ))

#port = mido.open_input()
#
#for msg in port:
#    print(msg)

#for msg in port:
#    t0 = time.time()
#    print(msg)
#    if msg.type == "note_on":
#        print("velocity = ", msg.velocity)
#t0 = time.time()

#while True:
#    for msg in port.iter_pending():
#        print(msg)
#        print(time.time() - t0)
#        #t0 = time.time()
        
        
#if time.time() - t0 < 20.0:
#    print(time.time() - t0)
#    for msg in port:
#        print(msg)
#        if msg.type == "note_on":
#            piano_dict[msg.note] = (x,y)
#            y+=0.0236667
#        print(piano_dict)
##        if msg.type == "set_tempo":
##            print(msg.tempo)
##        elif msg.type == "note_on":
##            if msg.velocity
##        if msg.type == "note_on":
##            print(msg.note)
##            print(msg.velocity)
#        #print(msg.velocity)
#else:
#    port.close()
#    print("end loop", piano_dict)
# get the list of all events
# events = mid.get_events()
# get the np array of piano roll image
#roll = mid.get_roll()
# draw piano roll by pyplot
#mid.draw_roll()
#Collapse