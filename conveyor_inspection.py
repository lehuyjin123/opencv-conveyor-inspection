"""Traditional conveyor inspection. No ML/DL. See README for assumptions.
Usage: python conveyor_inspection.py --input conveyor_inspection_student.mp4 --output result.mp4
"""
import argparse, csv, json
from queue import Queue, Empty, Full
from threading import Thread, Event
from time import perf_counter
from pathlib import Path
import cv2
import numpy as np



class VideoPipeline:
    """Ba luồng: đọc -> xử lý tuần tự -> ghi, hàng đợi giới hạn RAM.

    Tracking phụ thuộc thứ tự thời gian nên chỉ xử lý trên luồng chính.
    Mỗi VideoCapture/VideoWriter chỉ được sử dụng bởi một worker.
    Không bỏ frame khi đầy queue: chờ để giữ nguyên video và bộ đếm.
    """
    def __init__(self, cap, writer, queue_size=8):
        self.cap, self.writer = cap, writer
        self.inputs, self.outputs = Queue(queue_size), Queue(queue_size)
        self.stop = Event()
        self.errors = []
        self.workers = [Thread(target=self._read, daemon=True),
                        Thread(target=self._write, daemon=True)]
        for worker in self.workers:
            worker.start()

    def _put(self, queue, item):
        while not self.stop.is_set():
            try:
                queue.put(item, timeout=.1)
                return True
            except Full:
                pass
        return False

    def _get(self, queue):
        while not self.stop.is_set():
            try:
                return queue.get(timeout=.1)
            except Empty:
                pass
        self.check()
        return None

    def check(self):
        if self.errors:
            raise RuntimeError('Video I/O worker failed') from self.errors[0]

    def _read(self):
        try:
            while not self.stop.is_set():
                start = perf_counter()
                ok, frame = self.cap.read()
                read_ms = (perf_counter() - start) * 1000
                if not ok:
                    self._put(self.inputs, None)
                    break
                if not self._put(self.inputs, (frame, read_ms)):
                    break
        except BaseException as exc:
            self.errors.append(exc)
            self.stop.set()
        finally:
            self.cap.release()

    def _write(self):
        try:
            while not self.stop.is_set():
                frame = self._get(self.outputs)
                if frame is None:
                    break
                self.writer.write(frame)
        except BaseException as exc:
            self.errors.append(exc)
            self.stop.set()
        finally:
            self.writer.release()

    def read(self):
        self.check()
        return self._get(self.inputs)

    def write(self, frame):
        if not self._put(self.outputs, frame):
            self.check()
            raise RuntimeError('Video pipeline stopped')

    def close(self, abort=False):
        if abort:
            self.stop.set()
        else:
            # Sentinel đứng sau toàn bộ frame: ghi hết trước khi đóng file.
            self._put(self.outputs, None)
            self.workers[1].join()
            self.stop.set()
        for worker in self.workers:
            worker.join()
        self.check()


def detect(frame, roi_bounds=(400, 110, 575, 300)):
    """Trả về đặc trưng từng thẻ, mask và gamma của một khung hình."""
    # PHƯƠNG PHÁP 1 đã học: Median 5x5 loại nhiễu muối tiêu.
    # Chỉ xử lý pixel trong ROI, không phát hiện toàn băng tải rồi lọc vị trí.
    x0, y0, x1, y1 = roi_bounds
    roi = cv2.medianBlur(frame[y0:y1, x0:x1], 5)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value = hsv[:,:,2].copy()
    # PHƯƠNG PHÁP 2 đã học: Gamma trên V; giữ bản gốc để đo chấm đen.
    level = max(float(np.percentile(value, 90)), 1)
    gamma = float(np.clip(np.log(170/255)/np.log(min(level/255,.99)), .45, 1.8))
    lut = np.uint8(np.clip((np.arange(256)/255.)**gamma*255,0,255))
    hsv[:,:,2] = cv2.LUT(value,lut)
    # PHƯƠNG PHÁP 3 đã học: Color slicing bằng ngưỡng màu.
    # Giữ mọi Hue để thẻ xanh sai màu vẫn được phát hiện và tính FAIL.
    mask = cv2.inRange(hsv, (0,95,45), (179,255,255))
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    # Contour + hình chữ nhật xoay cho diện tích, kích thước và centroid.
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<650: continue
        rect=cv2.minAreaRect(contour); (cx,cy),(a,b),angle=rect
        if min(a,b)<15: continue
        x,y,w,h=cv2.boundingRect(contour)
        if x<2 or x+w>mask.shape[1]-2 or y<2 or y+h>mask.shape[0]-2: continue # Thẻ bị cắt ở mép chưa đủ điều kiện đo.
        cm=np.zeros(mask.shape,np.uint8); cv2.drawContours(cm,[contour],-1,255,-1)
        inner=cv2.erode(cm,np.ones((9,9),np.uint8))
        hue=float(np.median(hsv[:,:,0][mask.astype(bool)&(inner>0)]))
        # Ngưỡng tương đối trên V gốc đã lọc giúp tìm chấm cả khi đèn thay đổi.
        reference=float(np.median(value[(inner>0)&(mask>0)]))
        dark=np.uint8((value<reference*.43)&(inner>0))*255
        dots,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        dots=[d for d in dots if 15<cv2.contourArea(d)<220]
        offset=1.; dot_area=0.
        if dots:
            d=max(dots,key=cv2.contourArea); m=cv2.moments(d)
            if m['m00']:
                offset=float(np.hypot(m['m10']/m['m00']-cx,m['m01']/m['m00']-cy)/min(a,b))
                dot_area=float(m['m00'])
        box=cv2.boxPoints(rect); box += np.array([x0, y0], dtype=np.float32)
        found.append(dict(x=float(cx+x0),y=float(cy+y0),area=area,long=max(a,b),short=min(a,b),hue=hue,offset=offset,dot_area=dot_area,box=box.astype(int)))
    return found,mask,gamma


def verdict(samples):
    """Trung vị nhiều frame giảm tác động của chớp sáng và nhòe nhất thời."""
    med={k:float(np.median([s[k] for s in samples])) for k in ('area','long','short','hue','offset','dot_area')}
    reasons=[]
    if not 8<=med['hue']<=35: reasons.append('COLOR')
    if not (64<=med['long']<=85 and 40<=med['short']<=57): reasons.append('SIZE')
    if med['dot_area']<15: reasons.append('NO DOT')
    elif med['offset']>.20: reasons.append('DOT OFFSET')
    return ('FAIL' if reasons else 'PASS'), '+'.join(reasons) or 'OK', med


def run(args):
    cap=cv2.VideoCapture(str(args.input))
    if not cap.isOpened(): raise RuntimeError('Cannot open input video')
    fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
    if (w,h)!=(640,360): raise ValueError('ROI calibrated for 640x360 input')
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < args.line < x1 <= w and 0 <= y0 < y1 <= h):
        cap.release()
        raise ValueError('ROI must fit frame and contain counting line')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened(): raise RuntimeError('Cannot create output video')
    tracks={}; next_id=1; records=[]; frame_id=0; total=passed=failed=0
    timings = []
    wall_start = perf_counter()
    pipeline = VideoPipeline(cap, writer, args.queue_size) if not args.no_threads else None
    completed = False
    try:
      while True:
        iteration_start = perf_counter()
        if pipeline:
            item = pipeline.read()
            if item is None: break
            frame, read_ms = item
        else:
            read_start = perf_counter()
            ok,frame = cap.read()
            read_ms = (perf_counter()-read_start)*1000
            if not ok: break
        process_start = perf_counter()
        detections,mask,gamma=detect(frame, args.roi); used=set()
        for d in sorted(detections,key=lambda d:d['x']):
            candidates=[]
            for ident,t in tracks.items():
                gap=frame_id-t['last']
                if ident in used or gap>12: continue
                predicted=t['x']+t['vx']*gap
                distance=np.hypot(d['x']-predicted,d['y']-t['y'])
                if distance<max(28,9*gap): candidates.append((distance,ident))
            if candidates:
                ident=min(candidates)[1]; t=tracks[ident]
                gap=frame_id-t['last']; t['vx']=.7*t['vx']+.3*(d['x']-t['x'])/gap
            else:
                ident=next_id; next_id+=1
                t=dict(x=d['x'],y=d['y'],vx=5.,last=frame_id,samples=[],counted=False,start_x=d['x'])
                tracks[ident]=t
            used.add(ident); previous=t['x']; t.update(x=d['x'],y=d['y'],last=frame_id)
            t['samples'].append({k:v for k,v in d.items() if k!='box'})
            status,reason,features=verdict(t['samples'])
            # Chốt một lần khi tâm thẻ qua vạch; xử lý đúng thứ tự frame.
            if not t['counted'] and t['start_x']<args.line and d['x']>=args.line and len(t['samples'])>=3:
                t['counted']=True; t['decision']=(status,reason); total+=1
                passed+=status=='PASS'; failed+=status=='FAIL'
                records.append(dict(id=ident,time_s=round(frame_id/fps,3),status=status,reason=reason,**features))
            if t['counted']: status,reason=t['decision']
            color=(70,230,70) if status=='PASS' else (40,40,255)
            cv2.polylines(frame,[d['box']],True,color,2)
            cv2.circle(frame,(round(d['x']),round(d['y'])),3,color,-1)
            label=f'#{ident} {status} {reason}'
            xx=x0+4; yy=min(y1-8, max(y0+15, int(d['box'][:,1].min())-7))
            cv2.putText(frame,label,(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,.32,color,1,cv2.LINE_AA)
        cv2.rectangle(frame,(0,36),(640,99),(20,20,20),-1)
        cv2.putText(frame,f'TOTAL {total}    PASS {passed}    FAIL {failed}',(12,60),cv2.FONT_HERSHEY_SIMPLEX,.66,(255,255,255),2,cv2.LINE_AA)
        pending = sum(not t['counted'] for t in tracks.values())
        cv2.putText(frame,f'ROI only | Measured {len(tracks)} | Pending {pending} | Median + Gamma + Color',(12,83),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,220,230),1,cv2.LINE_AA)
        # ROI hiển thị cũng chính là vùng pixel truyền vào thuật toán.
        cv2.rectangle(frame,(x0,y0),(x1-1,y1-1),(255,220,0),2)
        cv2.line(frame,(args.line,y0+2),(args.line,y1-3),(255,255,0),1)
        cv2.putText(frame,'PROCESSING ROI',(x0+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,220,0),1,cv2.LINE_AA)
        # process_ms gồm phát hiện, tracking, phân loại và vẽ nhãn; không gồm I/O.
        process_ms = (perf_counter()-process_start)*1000
        submit_start = perf_counter()
        if pipeline:
            pipeline.write(frame)
        else:
            writer.write(frame)
        submit_ms = (perf_counter()-submit_start)*1000
        if args.show:
            cv2.imshow('Inspection',frame)
            key = cv2.waitKey(1)&255
        else:
            key = -1
        iteration_ms = (perf_counter()-iteration_start)*1000
        frame_id+=1
        timings.append(dict(frame=frame_id,video_time_s=(frame_id-1)/fps,read_ms=read_ms,process_ms=process_ms,submit_ms=submit_ms,iteration_ms=iteration_ms))
        if not args.quiet:
            print(f'Frame {frame_id:04d} | process={process_ms:.3f} ms | read={read_ms:.3f} ms | write/queue={submit_ms:.3f} ms | loop={iteration_ms:.3f} ms')
        if key == 27: break
      completed = True
    finally:
        if pipeline:
            pipeline.close(abort=not completed)
        else:
            cap.release(); writer.release()
    wall_seconds = perf_counter()-wall_start
    values = np.array([t['process_ms'] for t in timings])
    performance = dict(mode='sequential' if args.no_threads else 'threaded_io', wall_seconds=wall_seconds, effective_fps=frame_id/wall_seconds if wall_seconds else 0, mean_process_ms=float(values.mean()) if len(values) else 0, p95_process_ms=float(np.percentile(values,95)) if len(values) else 0, max_process_ms=float(values.max()) if len(values) else 0)
    with out.with_name(out.stem+'_timing.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['frame','video_time_s','read_ms','process_ms','submit_ms','iteration_ms']);wr.writeheader();wr.writerows(timings)
    if args.show: cv2.destroyAllWindows()
    with out.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['id','time_s','status','reason','area','long','short','hue','offset','dot_area']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(records)
    summary=dict(frames=frame_id,fps=fps,total=total,passed=passed,failed=failed,observed=len(tracks),not_crossed=sum(not t['counted'] for t in tracks.values()),records=records,performance=performance,roi=list(args.roi),count_line=args.line)
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='conveyor_inspection_student.mp4')
    p.add_argument('--output',default='result.mp4')
    p.add_argument('--line',type=int,default=480)
    p.add_argument('--roi',type=int,nargs=4,default=(400,110,575,300),metavar=('X0','Y0','X1','Y1'),help='Only process pixels within this rectangle')
    p.add_argument('--show',action='store_true',help='Requires opencv-python GUI build')
    p.add_argument('--no-threads', action='store_true', help='Sequential I/O for comparison')
    p.add_argument('--queue-size', type=int, default=8, help='Bounded I/O queue size')
    p.add_argument('--quiet', action='store_true', help='Suppress per-frame prints; still save timing CSV')
    args = p.parse_args()
    if args.queue_size < 1: p.error('--queue-size must be >= 1')
    run(args)



