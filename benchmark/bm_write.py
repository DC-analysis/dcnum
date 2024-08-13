"""Benchmark the writer thread"""
from dcnum import write


def main():

    thr_coll = write.QueueCollectorThread(
        data=self.dtin,
        event_queue=fe_kwargs["event_queue"],
        writer_dq=writer_dq,
        feat_nevents=fe_kwargs["feat_nevents"],
        write_threshold=500,
    )
    print("hello")
