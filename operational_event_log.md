# 2018-04-02
## Summary
TA1s (other than FiveDirections) were started back up after hours.  No benign data was generated.  A monitoring outage occurred in the evening, but no TA1s were interrupted.  

## Events
 * Monitoring outage
  * **Start time:** ~19:45 ET
  * **End time:** ~22:55 ET
  * **Impact:** All TA1s generated an alert as being down.  Graphs showed all TA1 publishing rates as 0.  We do not believe any TA1s actually lost data during this period.
  * **Cause:** Prometheus, which sits between the monitoring endpoints and the monitoring UI, went down.
  * **Resolution:** Prometheus was restarted.

# 2018-04-03
## Summary
FiveDirections was brought up prior to engagement hours beginning.  An operational issue resulted in the need to restart ClearScope and a new topic.  There were concerns about the THEIA run, so it was cleanly restarted and set to publish to a new topic.  Monitoring maintenance was performed to improve stability.  A rare data issue with FiveDirections data was found in the primary data stream, and a fix is being tested with the plan to deploy tomorrow morning.

## Events
 * ClearScope outage
  * **Start time:** ~09:25 ET
  * **End time:** ~10:05 ET
  * **Impact:** ClearScope stopped publishing.
  * **Cause:** An operational accident caused the phone to be disconnected from the publishing pipeline.
  * **Resolution:** The phone was reflashed, the topic was updated to ta1-clearscope-e3-official-1, and the phone began publishing a new stream.

 * THEIA restart
  * **Start time:** ~09:55 ET
  * **End time:** ~10:00 ET
  * **Impact:** THEIA stopped publishing.
  * **Cause:** We observed multiple instances of the relay-read-file process running on the target which may have been caused by skipping the reboot of the target during the startup procedure.  This was done to avoid reboots after TA5.1 staging.
  * **Resolution:** It was negotiated to do a reboot of the target and follow the standard startup procedure.  Following this, the topic was updated to ta1-theia-e3-official-1.

 * Monitoring maintenance
  * **Start time:** ~17:30 ET
  * **End time:** ~18:10 ET
  * **Impact:** The main cluster and test cluster prometheus servers were briefly brought down for maintenance, affecting monitoring graphs and alerting.
  * **Cause:** With new metrics being collected since the last engagement, we were finding that the prometheus processes occasionally ran out of memory and the processes crashed.
  * **Resolution:** We had plenty of memory to spare on the underlying servers, so we added more memory to the VMs and brought the services back up.

# 2018-04-04
## Summary
FiveDirections was restarted a couple of times to address issues, and each time a new topic was used.

## Events
 * FiveDirections data fix deployment
  * **Start time:** ~08:40 ET
  * **End time:** ~08:55 ET
  * **Impact:** Publishing was be stopped to the ta1-fivedirections-e3-official topic and started up to ta1-fivedirections-e3-official-1.
  * **Cause:** A rare issue in the data serialization of certain FileObject records caused deserialization failures due to syntax errors in the serialized data.  This issue was related to some information being missing based on when activity occurred on the system relative to when data collection was turned on.
  * **Resolution:** We deployed a fix for this issue on the translator machine.  We do not expect to see this particular issue again.

 * FiveDirections topic switch
  * **Start time:** ~09:50 ET
  * **End time:** ~10:00 ET
  * **Impact:** Publishing was be stopped to the ta1-fivedirections-e3-official-1 topic and started up to ta1-fivedirections-e3-official-2.
  * **Cause:** The ta1-fivedirections-e3-official-1 topic was somehow created by TA3 with 2 partitions.  We are still investigating why this happened, but multiple partitions in a topic causes record order to be lost across partitions.
  * **Resolution:** The publishing was switched to a new topic with only a single partition.


# 2018-04-05
## Summary
THEIA started encountering some performance issues, and needed to be patched and restarted a few times.  The end result was that the replay functionality will not be available starting with the data in the ta1-theia-e3-official-5 topic.

## Events
 * THEIA target machine applications unresponsive
  * **Start time:** ~10:30 ET
  * **End time:** ~11:45 ET
  * **Impact:** TA5.1 was unable to carry out any activity on the target machine.  A reboot of the host was required, affecting publishing.
  * **Cause:** Certain applications (sshd and sudo were observed) were unresponsive.  Because the set of applications included sudo, a reboot of the host was required in order to get things working again.
  * **Resolution:** A reboot occurred and resolved the issue with unresponsive applications.  Initially, publishing started to the topic ta1-theia-e3-official-2 by mistake.  Publishing was restarted to the ta1-theia-e3-official-3 topic, which is the stream that TA2s should consume from at this point.

 * THEIA target machine applications very slow
  * **Start time:** ~13:05 ET
  * **End time:** 2018-04-06 ~10:05 ET
  * **Impact:** TA5.1 was unable to carry out activity on the target machine in a timely manner, resulting in timeout issues with automated tools.
  * **Cause:** Certain applications (sshd was observed) were very slow.
  * **Resolution:** Given that this problem is similar to the previous observed issue, it was determined that a reboot was not enough, and a patch was needed.  The THEIA team patched the system and ran a stability test while publishing to the ta1-theia-e3-official-4 topic.  That topic contains corrupted data caused by an integration issue, and should be ignored.  In addition, we found that applying the patch did not resolve the issue.  As a final attempt, we will turn of the record/replay functionality for the target VM to see if it improves stability.  Publishing will begin to ta1-theia-e3-official-5.

# 2018-04-06
## Summary
CADETS experienced a crash shortly after noon, and the system was brought back up a little bit later.  FAROS is fixing a publishing issue causing PSB records to be lost.

## Events
 * The CADETS target system crashed
  * **Start time:** ~12:20 ET
  * **End time:** ~14:00 ET
  * **Impact:** The CADETS system stopped publishing.
  * **Cause:** Something caused the CADETS system to crash.  The cause of the crash is still unknown.
  * **Resolution:** The system was rebooted, and publishing began to the topic ta1-cadets-e3-official-1.

 * FAROS PSB publishing update
  * **Start time:** ~17:15 ET
  * **End time:** ~17:55 ET
  * **Impact:** The FAROS PSB stream was interrupted while the system was fixed.  Publishing will be started to the topic ta1-faros-e3-official-psb-1.
  * **Cause:** The way the system was set up resulted in only about half of the PSB records arriving at Kafka.
  * **Resolution:** The FAROS team provided fixes to the PSB system that are being deployed.

# 2018-04-09
## Summary
ClearScope was restored and published to a new topic after a device error occurred over the weekend.

 * ClearScope device error
  * **Start time:** 2018-04-08 ~13:10 ET
  * **End time:** ~09:55 ET
  * **Impact:** The ClearScope device had an error that caused it to stop functioning, including publishing data.
  * **Cause:** The cause is still unclear, but the problem was on the device itself.
  * **Resolution:** The ClearScope team has reflashed the phone and has set up publishing to the topic ta1-clearscope-e3-official-2.  TA5.1 interrupted publishing for around 30 minutes to re-stage the phone for the engagement after it was reflashed.

## Events

# 2018-04-10
## Summary
There were false positive alerts from monitoring overnight.  THEIA started publishing to a new topic.  FAROS started publishing to a new PSB topic.

## Events
 * Monitoring false positive alerts
  * **Start time:** 2018-04-09 ~19:00 ET, 2018-04-10 ~01:30 ET, 2018-04-10 ~03:30 ET
  * **End time:** 2018-04-09 ~19:00 ET, 2018-04-10 ~01:30 ET, 2018-04-10 ~03:30 ET
  * **Impact:** All TA1s were reported as not publishing briefly.
  * **Cause:** The cause was a timeout issue in the monitoring stack.
  * **Resolution:** The monitoring recovered on its own.  TA3 is investigating if there was a load issue around the times of the alerts which may have caused them.

 * FAROS false positive alerts
  * **Start time:** 2018-04-09 ~19:45 ET
  * **End time:** 2018-04-09 ~20:05 ET
  * **Impact:** Alerts were generated for FAROS.
  * **Cause:** These alerts were based on the DIFT topic, which is only populated on demand.  In general, there is no expected state for the publishing rate for this topic.
  * **Resolution:** TA3 will investigate removing alerts for this topic.

 * TRACE false positive alerts
  * **Start time:** 2018-04-09 ~21:15 ET
  * **End time:** 2018-04-09 ~23:00 ET
  * **Impact:** Alerts were generated for TRACE.
  * **Cause:** These alerts seem like they were caused by occasional low publishing rates that TRACE experiences under rare conditions.
  * **Resolution:** TA3 will investigate whether or not to tune the alert to be more accepting.  It may be the case that this condition is rare enough that the alert should be left alone in order to more quickly identify real problems.

 * THEIA target system unresponsive
  * **Start time:** ~11:25 ET
  * **End time:** ~12:50 ET
  * **Impact:** THEIA's target system became unresponsive, resulting in a manual reboot and topic change.
  * **Cause:** The reason for the unresponsiveness is still unknown.
  * **Resolution:** The system was rebooted, and publishing was started to the new topic ta1-theia-e3-official-6.  **UPDATE:** THEIA did not finish publishing the data they had gathered to the old ta1-theia-e3-official-5 topic, but they still have that data in tact.  They will be publishing it in parallel to a new topic ta1-theia-e3-official-5r.

 * FAROS target system rebooted to address slowness
  * **Start time:** ~14:00 ET
  * **End time:** ~14:15 ET
  * **Impact:** The FAROS target system was extremely slow, so a reboot was attempted.
  * **Cause:** The reason for the slowness is unknown currently.
  * **Resolution:** The system was rebooted, and publishing of the PSB data stream was started up to the new topic ta1-faros-e3-official-psb-2.

# 2018-04-11
## Summary
There were alerting false positives overnight.  THEIA restarted their publishing to a new topic.

## Events
 * Alerting false positives
  * **Start time:** 2018-04-10 ~17:55 ET, 2018-04-10 ~18:10 ET, and 2018-04-10 ~11:10 ET
  * **End time:** Shortly after each start time
  * **Impact:** All TA1s experience alerts overnight.  For some alerts, they never generated "OK" events.
  * **Cause:** There were timeouts between Grafana and Prometheus, likely due to a resource constraint.  We are investigating the cause still.
  * **Resolution:** The alerts were false positives, and all systems are working fine.

 * THEIA publishing restart
  * **Start time:** 2018-04-10 ~22:30 ET
  * **End time:** 00:55 ET
  * **Impact:** THEIA stopped publishing to the ta1-theia-e3-official-6 topic.  There was a false "OK" alert generated on 2018-04-10 at ~22:50 ET due to some publishing beginning to an incorrect topic ta1-theia-e3-offical-6r (note that "offical" was a typo present in the topic name).  THEIA also seems to be doing something with their old data and new topics.  See https://github.com/jallen89/theia_cdm18_samples#engagement-3-topics for details.
  * **Cause:** THEIA swapped out their publisher to resolve data and performance issues.
  * **Resolution:** Publishing is currently going to the ta1-theia-e3-official-6r topic.

 * CADETS crash
  * **Start time:** ~15:20 ET
  * **End time:** ~16:40 ET
  * **Impact:** The CADETS system crashed and stopped publishing.
  * **Cause:** The cause of the crash is still unknown.
  * **Resolution:** The CADETS system was rebooted, and publishing was started to the new topic ta1-cadets-e3-official-2.

# 2018-04-12
## Summary
Nothing to report.

# 2018-04-13
## Summary
FiveDirections and TRACE were rebooted in the morning.  All TA1s were shut down after 17:00 ET.

## Events
 * FiveDirections reboot
  * **Start time:** ~09:30 ET
  * **End time:** ~09:45 ET
  * **Impact:** The FiveDirections target was rebooted and it began publishing to a new topic.  Some data from the previous night and this morning was lost due to publishing backlog, but nothing critical was in there.
  * **Cause:** TA5.1 requested a reboot of the FiveDirections target.
  * **Resolution:** The target system was rebooted and publishing began to the new topic ta1-fivedirections-e3-official-3.

 * TRACE reboot
  * **Start time:** ~09:50 ET
  * **End time:** ~10:05 ET
  * **Impact:** The TRACE system has become very slow due to resource utilization, and a reboot was needed to recover.
  * **Cause:** The cause of the increased resource utilization is unknown, but it persisted from yesterday afternoon throughout the night.
  * **Resolution:** The TRACE system's publishing was cleanly shut down, and the system was rebooted.  Publishing began to a new topic ta1-trace-e3-official-1 after the reboot.
